import json
import os

from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from typing import Dict, Optional, List, Any

from deprecated.sphinx import deprecated
# from BCEmbedding import RerankerModel
# from BCERerank import BCERerank
from fastapi import Depends, Body
from langchain.chains.summarize.refine_prompts import prompt_template
# from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.callbacks import get_openai_callback

from langchain_core.messages import convert_to_openai_messages, SystemMessage, HumanMessage
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from db.milvus.milvus_client import connect_to_milvus
from dependencies.memory_dependency import get_memory_client
from llm.llm_provider import OpenAIProvider
from llm.state.planner_state import ReWOO

from utils.util import draw_lang_graph_flow

class PlanFormatter(BaseModel):
    plan: str = Field(description=f"""plans that are analyzed and broken down from a
                                    task and can solve the problem step by step""")
    tool: str = Field(description=f"""Actions of the plans,
     example: Google[input]: Worker that searches results from Google. Useful when you need to find short
    and succinct answers about a specific topic. The input should be a search query.
    """)
    step: str = Field(description="the sequence number of current step, example: Step#1, Step#2")
    query: str = Field(description="The instruction or the query")

class LlmApi:
    """
    This class is responsible for managing execution functions within lang graph nodes,
    each function accepts state and uses some props for its own logic
    """
    def __init__(self, model: str,
                 temperature: float,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.llm_base_url = base_url if base_url else os.environ.get('llm_base_url')
        self.api_key = api_key if api_key else os.environ.get('api_key')

        self.title_llm_url = os.environ.get('llm_base_url')
        self.title_api_key = os.environ.get('api_key')

        self.model_list = {
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

        for key in self.model_list.keys():
            if key in model:
                self.llm_base_url = self.model_list[key]['base_url']
                self.api_key = self.model_list[key]['api_key']

        self.provider = ChatOpenAI(base_url=self.llm_base_url, api_key=self.api_key, model=self.model)
        self.title_provider = ChatOpenAI(base_url=self.title_llm_url, api_key=self.title_api_key, model="qwen2:0.5b")

        self.memory_client = get_memory_client()


    @traceable
    def generate_conversation_title(self, state: dict, model: Optional[str] = "qwen2:0.5b") -> dict[str, Any]:
        """

        Args:
            state: langgraph state
            model: selected model, now using a 0.5b qwen model which can handle chinese for fast title generation

        Returns:
            state dict containing the generated title
        """
        user_message = state["message"][0].content
        system_template = f"""
                            You need to generate a title by using the input in 10 words.
                          """
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(system_template),
            ("human", user_message),
            ])
        try:
            title_generator = prompt_template| self.title_provider
            response = title_generator.invoke({})
            title = response.content
            return {'title': title}
        except Exception as e:
            raise e

    @deprecated(version="1.0", reason="Now first chat or continued chat classification has been merged based on the param provided")
    def create_first_chat(self, message: dict, model: Optional[str] = None) -> dict[str, dict]:
        model = model if model else self.model

        user_message = message["message"]
        system_template = "You are an assistant that helps with daily questions, coding"
        prompt_template = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_message}
        ]

        try:
            response = self.provider.get_response(prompt_template, model)
            return {"response": response}
        except Exception as e:
            raise e

    @traceable
    def chat(self, state: dict, model: Optional[str] = None) -> dict[str, dict]:
        """
        Args:
            state: langgraph state
            model: selected model

        Returns:
            state dict after adding the response from AI and extracted memory
        """
        model = model if model else self.model

        user_message = state["message"][0].content
        user_id = state["user_id"]
        conversation_id = state["conversation_id"]
        prompt_template = state["prompt_template"]

        try:
            response_generator = prompt_template | self.provider
            response = response_generator.invoke({})

            # store the memory
            conversation_memory = self.memory_client.add_memory_by_conversation_id(f"User: {user_message}\n Assistant: {response['message']}",
                                                                   conversation_id=conversation_id)
            user_memory = self.memory_client.add_memory_by_user_id(f"User: {user_message}\n",
                                                                    user_id=user_id)

            return {"response": response}
        except Exception as e:
            print(e)
            raise e

    def chat1(self, state: dict, model: Optional[str] = None) -> dict[str, dict]:
        """
        Args:
            state: langgraph state
            model: selected model

        Returns:
            state dict after adding the response from AI and extracted memory
        """
        model = model if model else self.model

        user_message = state["message"][0].content
        user_id = state["user_id"]
        history_messages = state["history_messages"]
        conversation_id = state["conversation_id"]

        system_template = f"""
                            You are an assistant that helps with daily questions, english, math and coding\n
                            You also may get memories or chat histories input,
                            please respond to the question from the user.\n
                            Input:\n
                            [USER_MEMORY_BEGIN] (if any)\n
                            some key user figures that can be helpful for this round of conversation\n
                            [USER_MEMORY_END] (if any)\n

                            [CONVERSATION_MEMORY_BEGIN] (if any)\n
                            some key inference from previous history messages that
                            can be helpful for this round of conversation\n
                            [CONVERSATION_MEMORY_END] (if any)\n

                            [HISTORY_BEGIN]\n
                            histories messages containing both user questions and your response\n
                            [HISTORY_END]\n
                            user's new query
                           """
        prompt_template = [
            {"role": "system", "content": system_template}
        ]
        try:
            # retrieve memory
            user_memory = self.memory_client.search_memory_by_user_id(user_message, user_id)
            if user_memory:
                memory_prompt = "Relevant user information from previous conversations:\n [USER_MEMORY_BEGIN]"
                for memory in user_memory:
                    memory_prompt += f"- {memory['memory']}\n"
                memory_prompt += "[USER_MEMORY_END]"
                prompt_template.append({"role": "user", "content": memory_prompt})

            conversation_memory = self.memory_client.search_memory_by_conversation_id(user_message,
                                                                                      conversation_id=conversation_id)
            if user_memory:
                memory_prompt = "Relevant user information from previous conversations:\n [USER_MEMORY_BEGIN]"
                for memory in user_memory:
                    memory_prompt += f"- {memory['memory']}\n"
                memory_prompt += "[USER_MEMORY_END]"
                prompt_template.append({"role": "user", "content": memory_prompt})

            if conversation_memory:
                memory_prompt = "Relevant key information from previous conversations:\n [CONVERSATION_MEMORY_BEGIN]"
                for memory in conversation_memory:
                    memory_prompt += f"- {memory['memory']}\n"
                memory_prompt += "[CONVERSATION_MEMORY_END]"
                prompt_template.append({"role": "user", "content": memory_prompt})

            if history_messages:
                prompt_template.append({"role": "user", "content": "full chat history records:\n [HISTORY_BEGIN]"})
                for history in history_messages:
                    history_message = {"role": history.role, "content": history.message}
                    prompt_template.append(history_message)
                prompt_template.append({"role": "user", "content": "\n [HISTORY_END]"})

            prompt_template.append({"role": "user", "content": f"query: {user_message}"})

            response = self.provider.get_response(prompt_template, model)

            conversation_memory = self.memory_client.add_memory_by_conversation_id(
                f"User: {user_message}\n Assistant: {response['message']}",
                conversation_id=conversation_id)

            # store the memory
            user_memory = self.memory_client.add_memory_by_user_id(f"User: {user_message}\n",
                                                                   user_id=user_id)

            return {"response": response}
        except Exception as e:
            print(e)
            raise e

    @traceable
    def construct_prompt(self, state: dict) -> dict:
        user_memory = state.get("user_memory", None)
        conversation_memory = state.get("conversation_memory", None)
        user_message = state["message"][0].content
        history_messages = state.get("history_messages", None)
        web_search_result = state.get("web_search_result", None)
        retrieved_context = state.get("rag_context", None)

        system_template = f"""
                            You are an assistant that helps with daily questions, english, math and coding\n
                            You also may get memories or chat histories input, or web search result which is
                            related to the user's query.
                            please respond to the question from the user.\n
                            Input:\n
                            [USER_MEMORY_BEGIN] (if any)\n
                            some key user figures that can be helpful for this round of conversation\n
                            [USER_MEMORY_END] (if any)\n

                            [CONVERSATION_MEMORY_BEGIN] (if any)\n
                            some key inference from previous history messages that
                            can be helpful for this round of conversation\n
                            [CONVERSATION_MEMORY_END] (if any)\n

                            [HISTORY_BEGIN]\n
                            histories messages containing both user questions and your response\n
                            [HISTORY_END]\n
                            
                            [WEB_SEARCH_RESULT_BEGIN]\n
                            web search result messages containing information collected from search engine\n
                            [WEB_SEARCH_RESULT_END]\n
                            user's new query
                           """
        prompt_template = [SystemMessage(system_template)]

        # retrieve memory

        if user_memory:
            memory_prompt = "Relevant user information from previous conversations:\n [USER_MEMORY_BEGIN]"
            for memory in user_memory:
                memory_prompt += f"- {memory.content}\n"
            memory_prompt += "[USER_MEMORY_END]"
            prompt_template.append(HumanMessage(memory_prompt))

        if conversation_memory:
            memory_prompt = "Relevant key information from previous conversations:\n [CONVERSATION_MEMORY_BEGIN]"
            for memory in conversation_memory:
                memory_prompt += f"- {memory['memory']}\n"
            memory_prompt += "[CONVERSATION_MEMORY_END]"
            prompt_template.append(HumanMessage(memory_prompt))

        # chat history
        if history_messages:
            history_prompt = "full chat history records:\n [HISTORY_BEGIN]"
            prompt_template.append(("human", "full chat history records:\n [HISTORY_BEGIN]"))
            for history in history_messages:
                history_prompt += f"- role: {history.role} message: {history.message}\n"
                # history_message = ("human", history.message)
                # prompt_template.append(history_message)
            history_prompt += "\n [HISTORY_END]"
            prompt_template.append(HumanMessage(history_prompt))

        if web_search_result:
            web_message_prompt = "web search result:\n [WEB_SEARCH_RESULT_BEGIN]\n"
            for result in web_search_result:
                web_message_prompt += f"- {result['content']}\n"
            web_message_prompt += "[WEB_SEARCH_RESULT_END]\n"

            prompt_template.append(HumanMessage(web_message_prompt))

        if retrieved_context:
            retrieved_context_prompt = "retrieved context:\n [RETRIEVED_CONTEXT_BEGIN]\n"
            for document in retrieved_context:
                retrieved_context_prompt += f"- {document}\n"
            retrieved_context_prompt += "[RETRIEVED_CONTEXT_END]\n"

            prompt_template.append(HumanMessage(retrieved_context_prompt))

        # prompt_template.append(HumanMessage("query\n" + user_message))

        return {"prompt_template": prompt_template}

    @traceable
    def search_conversation_memory(self, state: dict) -> dict:
        conversation_memory = self.memory_client.search_memory_by_conversation_id(state["message"][0].content,
                                                                                  state["user_id"])

        return {"conversation_memory": conversation_memory}

    @traceable
    def add_conversation_memory(self, state: dict) -> dict:
        pass

    @traceable
    def search_user_memory(self, state: dict) -> dict:
        user_memory = self.memory_client.search_memory_by_user_id(state["message"][0].content, state["user_id"])
        memory_content = ""
        for memory in user_memory:
            memory_content += f"{memory['memory']}\n"
        user_memory_message = HumanMessage(memory_content)

        return {"user_memory": user_memory_message}

    @traceable
    def add_user_memory(self, state: dict) -> dict:
        pass

    @traceable
    def search_web(self, state: dict) -> dict:
        message = state["message"]
        tool = TavilySearchResults(
            max_results=5,
            include_answer=True,
            include_raw_content=True,
            include_images=True,
            # search_depth="advanced",
            # include_domains = []
            # exclude_domains = []
        )

        search_result = tool.invoke({'query': message})

        return {"web_search_result": search_result}

    def get_plan(self, state: dict) -> dict:
        # Initialize node history if not present
        if "node_history" not in state:
            state["node_history"] = []

        state["node_history"].append("plan")
        print(f"[get_plan] Current node history: {state['node_history']}")

        task = state["task"]
        prompt = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
        which external tool together with tool input to retrieve evidence.

        Tools can be one of the following:
        (1) Google[input]: Worker that searches results from Google. Useful when you need to find short
        and succinct answers about a specific topic. The input should be a search query.
        (2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
        world knowledge and common sense. Prioritize it when you are confident in solving the problem
        yourself. Input can be any instruction.

        Your response should be in JSON format with the following structure for each step:
        {{
            "plan": "Detailed description of the plan",
            "tool": "Tool",
            "step": "Step#N",
            "query": "Query strings"
        }}
        Do not wrap the json result in markdown format or json markers
        
        Example:
        [
            {{
                "plan": "Understand the concept of black holes, their formation, and properties to ensure foundational knowledge.",
                "tool": "Google",
                "step": "Step#1",
                "query": "Latest discoveries about black holes in astrophysics."
            }},
            {{
                "plan": "Research the latest discoveries or notable black hole studies to find up-to-date and relevant examples.",
                "tool": "LLM",
                "step": "Step#2",
                "query": "Explain the concept of black holes, their formation, and key properties."
            }},
            {{
                "plan": "Based on the collected information, summarize the findings and explain how black holes influence their surroundings.",
                "tool": "LLM",
                "step": "Step#3",
                "query": "Summarize the latest findings about black holes and their impact on their surroundings."
            }}
        ]


        Note: You can only call Google tool once in the plan.

        Task: {task}"""

        prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
        planner = prompt_template | self.provider
        result = planner.invoke({"task": task})

        # Parse the JSON response into PlanFormatter objects
        steps = []
        try:
            # Assuming the response is a list of JSON objects
            # plan_steps = json.loads(result.content.replace("```json", "").replace("```", "").strip())
            plan_steps = json.loads(result.content.strip())
            for step in plan_steps:
                formatted_step = PlanFormatter(**step)
                # Convert to tuple format expected by the rest of the code
                steps.append((
                    formatted_step.plan,
                    formatted_step.step,
                    formatted_step.tool.split('[')[0],  # Extract tool name
                    formatted_step.query  # Extract input
                ))
        except Exception as e:
            print(f"Error parsing plan: {e}")
            steps = []

        return {"steps": steps, "plan_string": result.content}

    @traceable
    def tool_execution(self, state: dict) -> dict:
        """Execute tools according to the plan and collect evidence for each step.
        
        Args:
            state: Contains steps, plan_string, and other context
            
        Returns:
            Dictionary containing execution results and evidence
        """
        state["node_history"].append("tool")
        print(f"[tool_execution] Current node history: {state['node_history']}")

        steps = state["steps"]
        evidence_store = {}  # Store evidence from each step
        final_results = []

        for step_num, (plan, step_id, tool, instruction) in enumerate(steps):
            # Replace any #E references in the instruction with actual evidence
            for prev_step in range(step_num):
                # instruction = instruction.replace(f"Step#{prev_step}",
                #                                evidence_store.get(f"Step#{prev_step}", ""))
                instruction = evidence_store.get(f"Step#{prev_step}", "")

            # Execute the appropriate tool
            if tool == "Google":
                search_tool = TavilySearchResults(max_results=5)
                search_results = search_tool.invoke({"query": instruction})
                evidence = [result['content'] for result in search_results]
                
            elif tool == "LLM":
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an AI assistant helping with a multi-step task."),
                    ("user", f"Plan: {plan}\nInstruction: {instruction}")
                ])
                chain = prompt | self.provider
                evidence = chain.invoke({})
                evidence = evidence.content if hasattr(evidence, 'content') else str(evidence)
            
            # Store evidence for this step
            evidence_key = f"Step#{step_num}"
            evidence_store[evidence_key] = evidence
            
            final_results.append({
                "plan": plan,
                "step": step_id,
                "tool": tool,
                "evidence": evidence
            })

        return {
            "results": final_results,
        }

    def solve(self, state: dict) -> dict:
        """
        Final node in the planner workflow that generates the solution
        """
        state["node_history"].append("solve")
        print(f"[solve] Current node history: {state['node_history']}")
        solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
            retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
            contain irrelevant information.

            {plan}

            Now solve the question or task according to provided Evidence above. Respond with the answer
            directly with no extra words.

            Task: {task}
            """


        steps = state["results"]
        plan = ""
        for _, step in enumerate(steps):
            _plan, step_id, tool, instruction = step['plan'], step['step'], step['tool'], step['evidence']
            plan += f"Plan: {_plan}\n{step_id} = {tool}[{instruction}]\n"
        prompt = solve_prompt.format(plan=plan, task=state["task"])
        prompt_template = state["prompt_template"]
        prompt_template.append(SystemMessage(prompt))
        try:

            result = self.provider.invoke(prompt_template)
            return {"result": result}
        except Exception as e:
            print(e)
            raise e

    @traceable
    def search_rag_context(self, state: dict) -> dict:
        """
        Perform a local vector store db similarity search and reranking using bge,
        if the reranked result scores are poorly evaluated, a web search will be involved in the next phase
        Args:
            state:

        Returns:

        """
        reranked_score_count = 0
        relevance_content = []
        call_web_search = False
        user_message = state["message"][0].content
        vector_client = connect_to_milvus()
        retrieved_docs = vector_client.search_similarity(query=user_message, k=10)
        if retrieved_docs:
            str_docs = []
            for doc in retrieved_docs:
                str_docs.append(doc.page_content)
            reranked_docs = vector_client.rerank_document(query=user_message, documents=str_docs)

            for result in reranked_docs:
                if result.score > 0.5:
                    reranked_score_count += 1
                relevance_content.append(result.text)

            if reranked_score_count <= 2:
                call_web_search = True
            return {
                "rag_context": relevance_content,
                "call_web_search": call_web_search
            }

        return {"rag_context": None,
                "call_web_search": call_web_search}




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




