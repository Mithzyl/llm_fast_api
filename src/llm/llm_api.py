import os
from datetime import datetime
from typing import Dict, Optional, List

# from BCEmbedding import RerankerModel
# from BCERerank import BCERerank
from fastapi import Depends, Body
from langchain.chains.summarize.refine_prompts import prompt_template
# from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.callbacks import get_openai_callback

from langchain_core.messages import convert_to_openai_messages
from langgraph.constants import END, START
from langgraph.graph import StateGraph


from llm.llm_provider import OpenAIProvider
from llm.llm_state import State

from utils.util import draw_lang_graph_flow


class LlmApi:
    def __init__(self, model: str, temperature: float, base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.llm_base_url = base_url if base_url else os.environ.get('llm_base_url')
        self.api_key = api_key if api_key else os.environ.get('api_key')

        self.title_llm_url = os.environ.get('llm_base_url')
        self.title_api_key = os.environ.get('api_key')

        self.model_configs = {
            # configure model here
            'gpt': {
                'base_url': os.environ.get('openai_chat_url'),
                'api_key': os.environ.get('OPENAI_API_KEY')
            },
            'gemma': {
                'base_url': os.environ.get('gemma_base_url'),
                'api_key': os.environ.get('GEMMA_API_KEY')
            },
            'deepseek': {
                'base_url': os.environ.get('deepseek_base_url'),
                'api_key': os.environ.get('DEEPSEEK_API_KEY')
            },
            'qwen': {
                'base_url': os.environ.get('llm_base_url'),
                'api_key': os.environ.get('api_key')
            },
            'llama': {
                'base_url': os.environ.get('llm_base_url'),
                'api_key': os.environ.get('api_key')
            },


        }

        for key in self.model_configs.keys():
            if key in model:
                self.llm_base_url = self.model_configs[key]['base_url']
                self.api_key = self.model_configs[key]['api_key']

        self.provider = OpenAIProvider(self.llm_base_url, api_key=self.api_key)
        self.title_provider = OpenAIProvider(base_url=self.title_llm_url, api_key=self.title_api_key)

        # define lang graph workflow
        self.graph = StateGraph(State)


    def create_input_node(self, input_prompt: List[dict]) -> dict[str, list[dict]]:
        # open_ai_messages = convert_to_openai_messages(input_prompt['message'])
        state = State(input_prompt)
        return input_prompt


    def build_first_chat_workflow(self):
        # try for lang graph with debugging first
        self.graph.add_node("create_input_node", self.create_input_node)
        self.graph.add_node("generate_conversation_title", self.generate_conversation_title)
        self.graph.add_node("generate_conversation_response", self.create_first_chat)

        self.graph.set_entry_point("create_input_node")

        self.graph.add_edge("create_input_node", "generate_conversation_title")
        self.graph.add_edge("create_input_node", "generate_conversation_response")
        self.graph.add_edge("generate_conversation_title", END)
        self.graph.add_edge("generate_conversation_response", END)


    def run_workflow(self, prompt_input):
        try:
            self.build_first_chat_workflow()
            graph = self.graph.compile()

            # draw_lang_graph_flow(graph)

            finish_state = graph.invoke({"message": prompt_input})

            return finish_state

        except Exception as e:
            print(e)


    def generate_conversation_title(self, message: dict, model: Optional[str] = "qwen2:0.5b") -> str:
        user_message = message["message"]
        system_template = f"""
                            You need to generate a title by using the input in 10 words.
                          """
        prompt_template = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_message}
        ]
        try:
            response = self.title_provider.get_response(prompt_template, model)
            title = response.get('message', user_message[:10 if len(user_message) > 10 else len(user_message)])
            state = State({'title': title})
            return state
        except Exception as e:
            raise e


    def create_first_chat(self, message: dict, model: Optional[str] = None) -> Dict[str, str]:
        model = model if model else self.model

        user_message = message["message"]
        system_template = "You are an assistant that helps with daily questions, coding"
        prompt_template = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_message}
        ]

        try:
            response = self.provider.get_response(prompt_template, model)
            return State({"response": response})
        except Exception as e:
            raise e

    def chat(self, message_history: list, message: dict, model: Optional[str] = None) -> Dict[str, str]:
        model = model if model else self.model

        user_message = message["message"]
        system_template = "You are an assistant that helps with daily questions, english teaching and coding"
        prompt_template = [
            {"role": "system", "content": system_template}
        ]

        try:
            for history in message_history:
                history_message = {"role": history.role, "content": history.message}
                prompt_template.append(history_message)

            prompt_template.append({"role": "user", "content": user_message})

            response = self.provider.get_response(prompt_template, model)
            return {"response": response}
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




