# RAG Performance Comparison 

Retrieval Augmented Generation (RAG) has become a mainstream in the AI world. Due to its importance, we decided to conduct a comparison between some of the most commum RAG configuration.

The aim is to evaluate the performance of some RAG approaches in order to better understand the technology and evolve our RAG applications.

The work is organized in the following way:

1. **Two highly performatic sentence transformer retrievers**
    - msmarco-bert-base-dot-v5: https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5
    - BAAI/bge-large-en-v1.5: https://huggingface.co/BAAI/bge-large-en-v1.5
2. **Two state-of-the-art LLM models**
   - gpt-4-turbo-preview
   - gpt-3.5-turbo-0125
3. **All four combinations were tested**
4. **The framework TrueLens (https://www.trulens.org/) was used to measure the performance regarding: Groundedness, Answer and Context Relevance**

The results are:

![](img/vanna-readme-diagram.png)
