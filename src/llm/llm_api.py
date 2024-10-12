from datetime import datetime
from typing import Dict, Optional


from fastapi import Depends, Body
from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.callbacks import get_openai_callback

from langchain_community.document_compressors import FlashrankRerank
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from models.dao.message_dao import MessageDao
from utils.util import generate_md5_id

from BCEmbedding.tools.langchain import BCERerank


class LlmApi:
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

    def create_first_chat(self, message: str, model: Optional[str] = None) -> Dict[str, str]:
        model = model if model else self.model
        llm = ChatOpenAI(model=model, temperature=self.temperature, openai_api_base='https://api.deepseek.com')
        # llm = ChatOpenAI(
        #     model='deepseek-chat',
        #     openai_api_key='key',
        #     openai_api_base='https://api.deepseek.com',
        #     max_tokens=1024
        # )
        system_template = "You are an assistant that helps with daily questions, english teaching and coding"
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', system_template),
            ('user', '{message}')
        ])
        parser = StrOutputParser()
        chain = prompt_template | llm | parser
        try:
            with get_openai_callback() as cb:
                message_id = generate_md5_id()
                ai_msg = chain.invoke({"message": message})
                time = datetime.now()
                response = {'message_id': message_id,
                            'message': ai_msg,
                            'role': 'assistant',
                            'prompt_token': cb.prompt_tokens,
                            'completion_token': cb.completion_tokens,
                            'total_token': cb.total_tokens,
                            'cost': cb.total_cost,
                            'create_time': time,
                            'model': model
                            }
                return response
        except Exception as e:
            raise e

    def chat(self, message_history: list, new_message: str, model: Optional[str] = None) -> Dict[str, str]:
        model = self.model or model
        llm = ChatOpenAI(model=model, temperature=self.temperature)

        system_template = "You are an assistant that helps with daily questions, english teaching and coding"
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', system_template),
            ('placeholder', '{conversation}'),
            ('user', '{message}')
        ])
        parser = StrOutputParser()
        chain = prompt_template | llm | parser

        try:
            with get_openai_callback() as cb:
                message_id = generate_md5_id()
                ai_msg = chain.invoke(
                    {
                        "conversation": [(message.role, message.message) for message in message_history],
                        "message": new_message
                    }
                )
                time = datetime.now()
                response = {'message_id': message_id,
                            'message': ai_msg,
                            'role': 'assistant',
                            'prompt_token': cb.prompt_tokens,
                            'completion_token': cb.completion_tokens,
                            'total_token': cb.total_tokens,
                            'cost': cb.total_cost,
                            'create_time': time,
                            'model': model
                            }
                return response
        except Exception as e:
            raise e

    def RAG_chat(self, vectorstore, query):
        model = self.model
        llm = ChatOpenAI(model=model, temperature=self.temperature, api_key='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')
        PROMPT_TEMPLATE = """
        Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
        Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        <context>
        {context}
        </context>

        <question>
        {question}
        </question>

        The response should be specific and use statistics or numbers when possible.

        Assistant:"""

        # Create a PromptTemplate instance with the defined template and input variables
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "question", "country"]
        )
        # Convert the vector store to a retriever
        retriever = vectorstore.as_retriever(search_kwargs={"score_threshold": 0.5})


        # Define a function to format the retrieved documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def inspect(state):
            for k, v in state.items():
                print(v)
            return state

        reranker_args = {'model': 'ms-marco-MultiBERT-L-12', 'top_n': 50}
        ranker = Ranker(model_name='ms-marco-MultiBERT-L-12')

        reranker_args = {'model': 'maidalun1020/bce-reranker-base_v1', 'top_n': 5, 'device': 'cpu'}  # 修改为本地路径
        reranker = BCERerank(**reranker_args)
        compressor = FlashrankRerank(client=reranker, **reranker_args)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        rag_chain = (
                {"context": compression_retriever | format_docs, "question": RunnablePassthrough(), "country": RunnablePassthrough()}
                | RunnableLambda(inspect)
                | prompt
                | llm
                | StrOutputParser()
        )

        # rag_chain.get_graph().print_ascii()

        # Invoke the RAG chain with a specific question and retrieve the response
        res = rag_chain.invoke(query)

        return res


def get_llm_api(model: str = 'gpt-4o-mini-2024-07-18', temperature: float = 0.9) -> LlmApi:
    return LlmApi(model=model, temperature=temperature)
