import os
from datetime import datetime
from typing import Dict, Optional

# from BCEmbedding import RerankerModel
# from BCERerank import BCERerank
from fastapi import Depends, Body
# from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.callbacks import get_openai_callback

from langchain_community.document_compressors import FlashrankRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from llm.llm_provider import OpenAIProvider
from models.dao.message_dao import MessageDao
from models.model.llm_message import llm_message
from utils.util import generate_md5_id



class LlmApi:
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

        # TODO: get model base url for oprai and local model
        self.llm_base_url= os.environ.get('llm_base_url')
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.client = OpenAIProvider(self.llm_base_url, api_key=self.api_key)

    def create_first_chat(self, message: str, model: Optional[str] = None) -> Dict[str, str]:
        model = model if model else self.model

        system_template = "You are an assistant that helps with daily questions, coding"
        prompt_template = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": message}
        ]

        try:
            return self.client.get_response(prompt_template, model)
        except Exception as e:
            raise e

    def chat(self, message_history: list, message: str, model: Optional[str] = None) -> Dict[str, str]:
        model = model if model else self.model

        system_template = "You are an assistant that helps with daily questions, english teaching and coding"
        prompt_template = [
            {"role": "system", "content": system_template}
        ]

        try:
            for history in message_history:
                history_message = {"role": history.role, "content": history.message}
                prompt_template.append(history_message)

            prompt_template.append({"role": "user", "content": message})


            return self.client.get_response(prompt_template, model)
        except Exception as e:
            raise e

    # def RAG_chat(self, vectorstore, query, searches):
    #     model = self.model
    #     llm = ChatOpenAI(model=model, temperature=self.temperature, api_key=[])
    #     PROMPT_TEMPLATE = """
    #     Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
    #     Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
    #     The response must restrictly follow the limitations, e.g. if a condition (e.g. country) is provided,
    #     only give answers with the given condition.
    #
    #
    #     <context>
    #     {context}
    #     </context>
    #
    #     <question>
    #     {question}
    #     </question>
    #     Sometimes the question is implicit to retrieve information from the context, if so, try to analyze the information in the context
    #     and see if it meets the goal.
    #
    #
    #     The response should be specific and use statistics or numbers when possible.
    #
    #     Assistant:"""
    #
    #     # Create a PromptTemplate instance with the defined template and input variables
    #     prompt = PromptTemplate(
    #         template=PROMPT_TEMPLATE, input_variables=["context", "question", "country"]
    #     )
    #     # Convert the vector store to a retriever
    #     retriever = vectorstore.as_retriever(search_kwargs={"score_threshold": 0.5,
    #                                                         "k": 50})
    #
    #
    #     # Define a function to format the retrieved documents
    #     def format_docs(docs):
    #         return "\n\n".join(doc for doc in docs.get('rerank_passages', []))
    #
    #     def format_rerank(docs):
    #         return "\n\n".join(doc.page_content for doc in docs)
    #
    #     def inspect(state):
    #         for k, v in state.items():
    #             print(v)
    #         return state
    #
    #     ranker_args = {'model': 'ms-marco-MultiBERT-L-12', 'top_n': 10}
    #     ranker = Ranker(model_name='ms-marco-MultiBERT-L-12')
    #
    #     reranker_args = {'top_n': 5, 'device': 'cpu'}  # 修改为本地路径
    #     reranker = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")
    #     bce_reranker = None
    #     compressor = FlashrankRerank(client=ranker, **ranker_args)
    #
    #     compression_retriever = ContextualCompressionRetriever(
    #         base_compressor=bce_reranker, base_retriever=retriever
    #     )
    #
    #     sentence_pairs = [[query, passage.page_content] for passage in searches]
    #     # scores = reranker.compute_score(sentence_pairs)
    #     # rerank_results = reranker.rerank(query, [passage.page_content for passage in searches])
    #
    #     # reranked_content = format_docs(rerank_results)
    #
    #
    #     rag_chain = (
    #             {"context": compression_retriever | format_rerank, "question": RunnablePassthrough()}
    #             | RunnableLambda(inspect)
    #             | prompt
    #             | llm
    #             | StrOutputParser()
    #     )
    #
    #     # rag_chain.get_graph().print_ascii()
    #
    #     # Invoke the RAG chain with a specific question and retrieve the response
    #     # res = rag_chain.invoke({"context": reranked_content, "question": query})
    #
    #     res = rag_chain.invoke(query)
    #     return res


def get_llm_api(model: str = 'qwen2:0.5b', temperature: float = 0.9) -> LlmApi:
    return LlmApi(model=model, temperature=temperature)
