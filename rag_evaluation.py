# import sys, os
# # BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # sys.path.insert(0, BASE)
# from sys import path as pylib
# pylib.append('/home/ubuntu/ml_projects/RAG')

from LLM_API import Llm, Adapter, Config, SplitType

import numpy as np
from trulens_eval import TruChain, Feedback, Huggingface, Tru
from trulens_eval.schema import FeedbackResult
from trulens_eval.feedback.provider import OpenAI
from trulens_eval.feedback import Groundedness

import torch



def rag_evalution(openai, context):
    # How good is the retrieval
    context_relevance = (
        Feedback(openai.qs_relevance_with_cot_reasons, name="Context Relevance")
            .on_input()
            .on(context)
            .aggregate(np.mean)
    )

    # How good is the answer given the query
    qa_relevance = Feedback(openai.qs_relevance_with_cot_reasons, name="Answer Relevance").on_input_output()

    # At what extent answer is related to the context retrieved
    # grounded = Groundedness(groundedness_provider=openai, summarize_provider=openai)
    grounded = Groundedness(groundedness_provider=openai)
    # Define a groundedness feedback function
    groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundness")
            .on(context.collect())  # collect context chunks into a list
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
    )

    feedbacks = [qa_relevance, context_relevance, groundedness]

    return feedbacks


def main():

    torch.cuda.empty_cache()

    evaluation_questions = [
        "Kirk mentioned first, second, and third-degree meta data, what did he mean by that??",
        "What was Kirk's advice to people building geospatial software today?",
        "What is unstructured data?",
        "Summarize this episode in 500 words or less",
    ]

    tru = Tru()
    tru.reset_database()

    # 1 - OpenAI gpt-3.5-turbo-0125 + msmarco-bert-base-dot-v5

    sentence_transformer = "sentence-transformers/msmarco-bert-base-dot-v5"


    llmApi = Adapter(config=Config(splitType=SplitType.SENTENCE, chunkSize=400, chunkOverlap=50),
                     sentence_transformer_model=sentence_transformer)
    llm = Llm(llmApi)

    llm.addDocumentsFromFolder(path="data/")

    chain = llmApi.generativeQuery(with_source=False, openai="gpt-3.5-turbo-0125")

    openai = OpenAI()
    # select context to be used in feedback. the location of context is app specific.
    from trulens_eval.app import App
    context = App.select_context(chain)
    feedbacks = rag_evalution(openai, context)
    tru_recorder = TruChain(chain,
                            app_id='OpenAI gpt-3.5-turbo-0125 + msmarco-bert-base-dot-v5',
                            feedbacks=feedbacks)

    for question in evaluation_questions:
        # Excution with evaluation
        with tru_recorder as recording:
            llm_response = chain.invoke(question)

    # 2 - OpenAI gpt-4-turbo-preview + msmarco-bert-base-dot-v5

    # retriever
    sentence_transformer = "sentence-transformers/msmarco-bert-base-dot-v5"

    llmApi = Adapter(config=Config(splitType=SplitType.SENTENCE, chunkSize=400, chunkOverlap=50),
                     sentence_transformer_model=sentence_transformer)
    llm = Llm(llmApi)
    llm.addDocumentsFromFolder(path="data/")
    chain = llmApi.generativeQuery(with_source=False, openai='gpt-4-turbo-preview')

    context = App.select_context(chain)
    feedbacks = rag_evalution(openai, context)

    tru_recorder = TruChain(chain,
                            app_id='OpenAI gpt-4-turbo-preview + msmarco-bert-base-dot-v5',
                            feedbacks=feedbacks)

    for question in evaluation_questions:
        # Excution with evaluation
        with tru_recorder as recording:
            llm_response = chain.invoke(question)

    # 3 - OpenAI gpt-3.5-turbo-0125 + BAAI/bge-large-en-v1.5

    sentence_transformer = "BAAI/bge-large-en-v1.5"

    llmApi = Adapter(config=Config(splitType=SplitType.SENTENCE, chunkSize=400, chunkOverlap=50),
                     sentence_transformer_model=sentence_transformer)
    llm = Llm(llmApi)

    llm.addDocumentsFromFolder(path="data/")

    chain = llmApi.generativeQuery(with_source=False, openai="gpt-3.5-turbo-0125")

    openai = OpenAI()
    # select context to be used in feedback. the location of context is app specific.
    from trulens_eval.app import App
    context = App.select_context(chain)
    feedbacks = rag_evalution(openai, context)
    tru_recorder = TruChain(chain,
                            app_id='OpenAI gpt-3.5-turbo-0125 + BAAI/bge-large-en-v1.5',
                            feedbacks=feedbacks)

    for question in evaluation_questions:
        # Excution with evaluation
        with tru_recorder as recording:
            llm_response = chain.invoke(question)

    # 4 - OpenAI gpt-4-turbo-preview + BAAI/bge-large-en-v1.5

    # retriever
    sentence_transformer = "BAAI/bge-large-en-v1.5"

    llmApi = Adapter(config=Config(splitType=SplitType.SENTENCE, chunkSize=400, chunkOverlap=50),
                     sentence_transformer_model=sentence_transformer)
    llm = Llm(llmApi)
    llm.addDocumentsFromFolder(path="data/")
    chain = llmApi.generativeQuery(with_source=False, openai='gpt-4-turbo-preview')

    context = App.select_context(chain)
    feedbacks = rag_evalution(openai, context)

    tru_recorder = TruChain(chain,
                            app_id='OpenAI gpt-4-turbo-preview + BAAI/bge-large-en-v1.5',
                            feedbacks=feedbacks)

    for question in evaluation_questions:
        # Excution with evaluation
        with tru_recorder as recording:
            llm_response = chain.invoke(question)

    tru.get_leaderboard(app_ids=[])
    tru.run_dashboard()  # open a local streamlit app to explore

    # tru.stop_dashboard() # stop if needed


main()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CUDA Error  changing env variable CUDA_VISIBLE_DEVICES after program start.  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# sudo rmmod nvidia_uvm
# sudo modprobe nvidia_uvm
