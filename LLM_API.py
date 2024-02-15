import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Any, Optional
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PyPDFDirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
import faiss
from transformers import AutoTokenizer

logging.basicConfig(filename='error.log',filemode='w', level=logging.INFO, format='%(asctime)s - [%(levelname)s] (%(threadName)-9s) %(message)s', )
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] (%(threadName)-9s) %(message)s', )


class HaystackStores(Enum):
    FAISS = 1,
    OPENSEARCH = 2


class SplitType(Enum):
    CHARACTER = 1,
    WORD = 2,
    SENTENCE = 3,
    PARAGRAPH = 4


class Config:

    def __init__(self, splitType: SplitType=SplitType.WORD, chunkSize=200, chunkOverlap=50, contextWindow=4096, numOutput=256, index_name='faiss_index.faiss', top_k=5):
        self.chunk_size = chunkSize # The size of the text chunk (in characters not token) for a node . Is used for the node parser when they aren’t provided.
        self.chunk_overlap = chunkOverlap  # The amount of overlap between nodes (in characters).
        # The size of the context window of the LLM. Typically we set this automatically with the model metadata. But we also allow explicit override via this parameter for additional control (or in case the default is not available for certain latest models)
        self.context_window = contextWindow
        # The number of maximum output from the LLM. Typically we set this automatically given the model metadata. This parameter does not actually limit the model output, it affects the amount of “space” we save for the output, when computing available context window size for packing text from retrieved Nodes.
        self.num_output = numOutput
        self.chunk_overlap_ratio = 0.1
        self.chunk_size_limit = 3000
        self.splitType = splitType
        self.faiss_index_path = index_name
        self.top_k_for_retriever = top_k


class FrameworkAdapter(ABC):

    def __init__(self, config : Config, store, sentence_transformer_model:str, llm_model:str):
        self.document_store = store
        self.config = config
        self.sentence_transformer_model = sentence_transformer_model
        self.llm_model = llm_model

    '''
       Pre process, split to create chunks, initialize index database, insert into the vector store and in the document store.
       The answer always need to be able to return the actual documents used to produce the response (not only the context).
       '''
    @abstractmethod
    def getDocumentStore(self):
        return self.document_store

    @abstractmethod
    def getConfig(self):
        return self.config

    @abstractmethod
    def addDocument(self, docs:List, preprocessor):
        raise NotImplementedError()

    '''
    GenerativeQAPipeline combines the Retriever with the Generator. To create an Answer, the Generator uses the internal factual knowledge stored in the language model’s parameters and the external knowledge provided by the Retriever’s output.
    '''
    @abstractmethod
    def generativeQuery(self, model_path:str, with_source:bool):
        raise NotImplementedError()

    @abstractmethod
    def searchDocuments(self, query: List[str], retriever):
        raise NotImplementedError()

    @abstractmethod
    def getDocumentsFromFolder(self, path: str):
        raise NotImplementedError()




class Adapter(FrameworkAdapter):

    logger =  logging #logging.getLogger("haystack").setLevel(logging.INFO)

    def __init__(self,  config : Config, sentence_transformer_model:str, llm_model:str=None, store = None):

        Adapter.logger.info("Initializing HaystackAdapter")

        '''
         model_name = "sentence-transformers/all-mpnet-base-v2"
         model_kwargs = {'device': 'cpu'}
         encode_kwargs = {'normalize_embeddings': False}
         hf = HuggingFaceEmbeddings(
             model_name=model_name,
             model_kwargs=model_kwargs,
             encode_kwargs=encode_kwargs
         )                
         '''
        self.embeddings = HuggingFaceEmbeddings(model_name=sentence_transformer_model,
                                                model_kwargs={'device': 'cuda:0'})
        self.embedding_dim = self.embeddings.client.get_sentence_embedding_dimension()

        if store == None:
            if os.path.exists(config.faiss_index_path):
                Adapter.logger.info("Using index path " + config.faiss_index_path)
                store = FAISS.load_local(config.faiss_index_path, self.embeddings)
            else:
                Adapter.logger.info("No index found")

                # Another way to initialize the Faiss index store
                #texts = ["FAISS is an important library", "LangChain supports FAISS"]
                #store = FAISS.from_texts(texts, self.embeddings)

                index = faiss.IndexFlatL2(self.embedding_dim)
                store = FAISS(self.embeddings, index, InMemoryDocstore({}), {})


        super().__init__(config, store, sentence_transformer_model, llm_model)

        self.retriever = self._getEmbeddingRetriever()

        #self.embedding_dim = self.retriever.embedding_encoder.embedding_model[1].word_embedding_dimension


    def getDocumentStore(self):
        return self.document_store

    def getConfig(self):
        return self.config

    def getDocumentsFromFolder(self, path : str) -> List[Document]:

        file_name = None
        if os.path.isfile(path):
            file_name = os.path.split(path)[-1]
            path = os.path.abspath(path)
            path = "".join(os.path.split(path)[0:-1])

        # First, try to load TXT files
        text_loader_kwargs = {'autodetect_encoding': True}
        loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs, use_multithreading=True, show_progress=True)
        docs = loader.load()
        #Then, try to load PDF files
        loader = PyPDFDirectoryLoader(path, recursive=True)
        docs.extend(loader.load())

        # TODO - found a way to select only a specif file
        # if file_name != None:
        #     for doc in docs:
        #         if file_name not in doc:
        #             docs.remove(doc)

        return docs


    def _getDefaultPreprocessor(self):

        ''' lean_whitespace – Strip whitespaces before or after each line in the text.
            clean_header_footer – Use heuristic to remove footers and headers across different pages by searching for the longest common string. This heuristic uses exact matches and therefore works well for footers like "Copyright 2019 by XXX", but won't detect "Page 3 of 4" or similar.
            clean_empty_lines – Remove more than two empty lines in the text.
            remove_substrings – Remove specified substrings from the text. If no value is provided an empty list is created by default.
            split_by – Unit for splitting the document. Can be "word", "sentence", or "passage". Set to None to disable splitting.
            split_length – Max. number of the above split unit (e.g. words) that are allowed in one document. For instance, if n -> 10 & split_by -> "sentence", then each output document will have 10 sentences.
            split_overlap – Word overlap between two adjacent documents after a split. Setting this to a positive number essentially enables the sliding window approach. For example, if split_by -> `word`, split_length -> 5 & split_overlap -> 2, then the splits would be like: [w1 w2 w3 w4 w5, w4 w5 w6 w7 w8, w7 w8 w10 w11 w12]. Set the value to 0 to ensure there is no overlap among the documents after splitting.
            split_respect_sentence_boundary – Whether to split in partial sentences if split_by -> `word`. If set to True, the individual split will always have complete sentences & the number of words will be = split_length.
            tokenizer_model_folder – Path to the folder containing the NTLK PunktSentenceTokenizer models, if loading a model from a local path. Leave empty otherwise.
            language – The language used by "nltk.tokenize.sent_tokenize" in iso639 format. Available options: "ru","sl","es","sv","tr","cs","da","nl","en","et","fi","fr","de","el","it","no","pl","pt","ml"
            id_hash_keys – Generate the document id from a custom list of strings that refer to the document's attributes. If you want to ensure you don't have duplicate documents in your DocumentStore but texts are not unique, you can modify the metadata and pass e.g. `"meta"` to this field (e.g. [`"content"`, `"meta"`]). In this case the id will be generated by using the content and the defined metadata.
            progress_bar – Whether to show a progress bar.
            add_page_number – Add the number of the page a paragraph occurs in to the Document's meta field `"page"`. Page boundaries are determined by `"\f"` character which is added in between pages by `PDFToTextConverter`, `TikaConverter`, `ParsrConverter` and `AzureConverter`.
            max_chars_check – the maximum length a document is expected to have. Each document that is longer than max_chars_check in characters after pre-processing will raise a warning and is going to be split at the `max_char_check`-th char, regardless of any other constraint. If the resulting documents are still too long, they'll be cut again until all fragments are below the maximum allowed length.
        '''

        splitBy = ''
        splitLength = 0
        splitOverlap = 0
        respectSentenceBoundery=False

        if (self.config.splitType == SplitType.WORD):
            respectSentenceBoundery=True
            splitBy = "word"
            splitLength = 100 #self.config.chunk_size // 5  # 1000/10 -> 100 words
            splitOverlap = 10 #self.config.chunk_overlap // 5  # 20 / 5 -> 4 words
        elif (self.config.splitType == SplitType.CHARACTER):
            respectSentenceBoundery=True
            splitBy = "word"  # hay stack does not support split by character
            splitLength = 100 #self.config.chunk_size // 5  # 1000/10 -> 100 words
            splitOverlap = 10 #self.config.chunk_overlap // 5  # 20 / 5 -> 4 words
        elif (self.config.splitType == SplitType.SENTENCE):
            splitBy = "sentence"
            splitLength = 2 #self.config.chunk_size // 200  # 1000/200 -> 5 sentences
            splitOverlap = 1
        elif (self.config.splitType == SplitType.PARAGRAPH):
            splitBy = "passage"
            splitLength = 1 #self.config.chunk_size // 1000  # 1000/1000 -> 1 passage
            splitOverlap = 0  # 20 / 20 -> 1 sentence

        # return PreProcessor(
        #     clean_empty_lines=True,
        #     clean_whitespace=True,
        #     clean_header_footer=False,
        #     split_by=splitBy,
        #     split_length=splitLength,
        #     split_overlap=splitOverlap,
        #     split_respect_sentence_boundary=respectSentenceBoundery,
        #     language="pt"
        # )

        # from transformers import GPT2TokenizerFast, BertTokenizer
        # tokenizer = BertTokenizer.from_pretrained(self.sentence_transformer_model)
        # better this way
        tokenizer = AutoTokenizer.from_pretrained(self.sentence_transformer_model, use_fast=True)

        # OBS: CharacterTextSplitter does not make use of chunk_size and overlap
        # It is poorly implemented and should be avoided
        from langchain.text_splitter import CharacterTextSplitter
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)

        return text_splitter

    def _getRecursiveTextSplitter(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            add_start_index=True,
            separators=['.', '?', '!']
        )

        return text_splitter

    def _getEmbeddingRetriever(self):
        #vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        #retriever = self.document_store.as_retriever(search_type="similarity", search_kwargs={"k": self.config.top_k_for_retriever})
        #retriever = self.document_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": self.config.top_k_for_retriever,'score_threshold':0.7})
        retriever = self.document_store.as_retriever(search_type="mmr", # Maximum marginal relevance retrieval
                                                     search_kwargs={"k": self.config.top_k_for_retriever,})
        return retriever

    def addDocument(self, documents:List, preprocessor=None):

        if preprocessor == None:
             preprocessor=self._getDefaultPreprocessor()

        import shutil

        try:
            if os.path.isfile(self.config.faiss_index_path):
                os.remove(self.config.faiss_index_path)

            if os.path.isdir(self.config.faiss_index_path):
                shutil.rmtree(self.config.faiss_index_path)
        except:
            pass

        index = faiss.IndexFlatL2(self.embedding_dim)
        self.document_store = FAISS(self.embeddings, index, InMemoryDocstore({}), {})

        # OBS: CharacterTextSplitter does not make use of chunk_size and overlap
        # It is poorly implemented and should be avoided
        # split_doc = preprocessor.split_documents(documents)

        split_doc = self._getRecursiveTextSplitter().split_documents(documents)

        # insert
        db = self.getDocumentStore().from_documents(split_doc, self.embeddings)
        db.save_local(self.config.faiss_index_path)

        # reload document store
        self.document_store = FAISS.load_local(self.config.faiss_index_path, self.embeddings)
        self.retriever = self._getEmbeddingRetriever()

    def searchDocuments(self, query:str, rank:int=5, retriever=None) -> List[Document]:

        if retriever != None:
            return retriever.get_relevant_documents(query, ) # in this case, k is akready specified in the retriever
        else:
            return self.document_store.similarity_search_with_score(query, k=rank)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)


    def generativeQuery(self, model_path=None, with_source=True, openai='gpt-3.5-turbo-1106'):

        template = """Use the following pieces of context to answer the question at the end.
           If you don't know the answer, just say that you don't know, don't try to make up an answer.
           Keep the answer as concise as possible.            
           {context}
           Question: {question}
           Helpful Answer:"""
        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate.from_template(template)

        if model_path==None or self.llm_model==None:

            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model_name=openai, temperature=0)

            from langchain.schema import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough

            if with_source:

                rag_chain_from_docs = (
                        RunnablePassthrough.assign(context=(lambda x: self.format_docs(x["context"])))
                        | prompt
                        | llm
                        | StrOutputParser()
                )
                from langchain_core.runnables import RunnableParallel
                # If we want the docs being returned
                rag_chain_with_source = RunnableParallel(
                    {"context": self.retriever, "question": RunnablePassthrough()}).assign(answer=rag_chain_from_docs)

                #result = rag_chain_with_source.invoke(query)
                #print(result)

                return rag_chain_with_source

            else:
                # Without sources
                rag_chain = (
                        {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                )
                return rag_chain


        else:

            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = model_path if model_path != None else self.llm_model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
            hf = HuggingFacePipeline(pipeline=pipe)
            #result = pipe(template.format(context=self.searchDocuments(query), question=query))
            chain = prompt | hf
            #result = chain.invoke({"context": self.retriever | self.format_docs, "question": query} )
            #print(result)

            return chain




class Llm:

    logger = logging #logging.getLogger("LLM").setLevel(logging.INFO)

    def __init__(self, frameworkWrapper: FrameworkAdapter):
        Llm.logger.info("Initiating llm class.")
        self.framework = frameworkWrapper

    def addDocumentsFromFolder(self,  path:str):
        documents = self.framework.getDocumentsFromFolder(path=path)
        self.framework.addDocument(documents=documents)

    def semanticSearch(self, query:str, rank:int) -> List[Any]:
        result = self.framework.searchDocuments(query=query,rank=rank)
        return result

    def removeFaissIndex(self, path:str):
        os.remove(path)

    def generativeQAWithSource(self, query: str, model_path: str = None):
        chain = self.framework.generativeQuery(model_path, with_source=True)
        try:
            result = chain.invoke({"context": self.framework.retriever | self.framework.format_docs, "question": query})
        except:
            result = chain.invoke(query)

        return result

    def generativeQAWithoutSource(self, query: str, model_path: str = None):
        chain = self.framework.generativeQuery(model_path, with_source=False)
        try:
            result = chain.invoke({"context": self.framework.retriever | self.framework.format_docs, "question": query})
        except:
            result = chain.invoke(query)

        return result

