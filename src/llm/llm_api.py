import json
import os
from os import environ

import tiktoken
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
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
from llm.llm_factory import get_llm_provider
from llm.llm_provider import OpenAIProvider
from llm.state.planner_state import ReWOO

from utils.util import draw_lang_graph_flow
from llm.mcp.mcp_tool_manager import MCPToolManager


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
                 mcp_tool_manager: MCPToolManager,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model_config_file: Optional[str] = '/src/model_url_config.json',
                 ):
        self.model = model
        self.temperature = temperature
        self.mcp_tool_manager = mcp_tool_manager

        self.provider = get_llm_provider(
            model=self.model,
            temperature=self.temperature,
            base_url=base_url,
            api_key=api_key,
            model_config_file=model_config_file
        )
        self.title_provider = get_llm_provider(model="qwen3:0.6b", temperature=0.9)

        self.memory_client = get_memory_client()

        self.tokenizer = tiktoken.get_encoding("o200k_base")

        # Init tools from MCPToolManager
        self.tools = self.mcp_tool_manager.tools
        self.tool_map = self.mcp_tool_manager.tool_map


    def _format_tools_for_prompt(self) -> str:
        """
        Format tool list to be strings
        Returns:

        """
        tool_strings = []
        for i, tool in enumerate(self.tools):
            tool_strings.append(f"[{i+1}]: {tool.name}[input]: {tool.description}")
        return "\n".join(tool_strings)


    @traceable
    def generate_conversation_title(self, state: dict, model: Optional[str] = "qwen2:0.5b") -> dict[str, Any]:
        """

        Args:
            state: langgraph state
            model: selected model, now using a 0.5b qwen model which can handle chinese for fast title generation

        Returns:
            state dict containing the generated title
        """
        user_message = state["task"]
        system_template = f"""
                            You need to generate a title by using the input in 10 words as a sentence.
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


    @traceable
    def construct_prompt(self, state: dict) -> dict:
        user_memory = state.get("user_memory", None)
        conversation_memory = state.get("conversation_memory", None)
        user_message = state["task"]
        history_messages = state.get("history_messages", None)
        web_search_result = state.get("web_search_result", None)
        retrieved_context = state.get("rag_context", None)
        tool_results = state.get("results", None)

        plan_string = ""
        if tool_results:
            plan_string += "[TOOL_RESULTS_BEGIN]\n"
            for _, step in enumerate(tool_results):
                _plan, step_id, tool, instruction = step['plan'], step['step'], step['tool'], step['evidence']
                plan_string += f"Plan: {_plan}\n{step_id} = {tool}[{instruction}]\n"
            plan_string += "[TOOL_RESULTS_END]\n"

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

                            {plan_string}
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

        prompt_template.append(HumanMessage("query\n" + user_message))

        return {"prompt_template": prompt_template}

    @traceable
    def search_conversation_memory(self, state: dict) -> dict:
        conversation_memory = self.memory_client.search_memory_by_conversation_id(state["message"][0].content,
                                                                                  state["user_id"])

        return {"conversation_memory": conversation_memory}

    @traceable
    def add_conversation_memory(self, state: dict) -> dict:
        assistant_response = state.get("result", [])
        conversation_id = state.get("conversation_id")
        human_prompt = state.get("task")
        if assistant_response and conversation_id and human_prompt:
            final_response_content = assistant_response[-1].content
            self.memory_client.add_memory_by_conversation_id(
                f"User: {human_prompt}\nAssistant: {final_response_content}",
                conversation_id=conversation_id
            )
        return {}

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
        assistant_response = state.get("result", [])
        user_id = state.get("user_id")
        human_prompt = state.get("task")
        if assistant_response and user_id and human_prompt:
            final_response_content = assistant_response[-1].content
            self.memory_client.add_memory_by_user_id(
                f"User: {human_prompt}\nAssistant: {final_response_content}",
                user_id=user_id
            )
        return {}

    @traceable
    def prompt_rewrite(self, state: dict) -> dict:
        """
        Rewrite user prompt to be more clear and effective for LLM processing
        Args:
            state: Contains original message in state["message"]

        Returns:
            dict: Contains rewritten prompt in state["rewritten_prompt"]
        """
        message = state["message"]
        rewriter_prompt = f"""
        You are an expert at rewriting prompts to be more effective for large language models.
        Your task is to improve the following prompt while maintaining its original intent:

        Original Prompt: {message}

        Guidelines for rewriting:
        1. Clarify ambiguous terms or requests
        2. Add relevant context if needed
        3. Structure the prompt for better comprehension
        4. Keep technical terms precise
        5. Maintain original tone and intent
        6. If the prompt is already well-structured, return it unchanged

        Rewrite the prompt following these guidelines. Respond with just the rewritten prompt,
        no additional commentary or formatting.
        """
        prompt_template = ChatPromptTemplate.from_messages([("user", rewriter_prompt)])
        rewriter = prompt_template | self.provider
        rewritten = rewriter.invoke({})
        return {"rewritten_prompt": rewritten.content}

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

    async def get_plan(self, state: dict) -> dict:
        """
        Invokes the 'get_plan' MCP tool.
        """
        task = state["task"]
        get_plan_tool = self.tool_map.get("get_plan")
        if not get_plan_tool:
            raise ValueError("MCP tool 'get_plan' not found in tool_map.")

        # Serialize the list of tools to a JSON string
        tools_for_prompt = [{"name": tool.name, "description": tool.description} for tool in self.tools]
        tools_json = json.dumps(tools_for_prompt)

        # The MCP tool returns a JSON string, which we need to parse.
        plan_string = await get_plan_tool.ainvoke({"task": task, "tools_json": tools_json})
        try:
            plan_steps = json.loads(plan_string)
            steps = []
            for step in plan_steps:
                formatted_step = PlanFormatter(**step)
                steps.append((
                    formatted_step.plan,
                    formatted_step.step,
                    formatted_step.tool.split('[')[0].strip(),
                    formatted_step.query
                ))
            return {"steps": steps, "plan_string": plan_string}
        except json.JSONDecodeError as e:
            print(f"Error parsing plan from MCP tool: {e}")
            return {"steps": [], "plan_string": "[]"}

    @traceable
    async def tool_execution(self, state: dict) -> dict:
        """
        Executes the plan's steps locally using the tool_map.
        """
        steps = state["steps"]
        results = []
        for step in steps:
            plan, step_id, tool_name, query = step
            if tool_name in self.tool_map:
                tool_to_use = self.tool_map[tool_name]
                try:
                    # Assuming tools are synchronous for now
                    evidence = await tool_to_use.ainvoke({"query": query})
                    results.append({
                        "plan": plan,
                        "step": step_id,
                        "tool": tool_name,
                        "evidence": evidence
                    })
                except Exception as e:
                    results.append({
                        "plan": plan,
                        "step": step_id,
                        "tool": tool_name,
                        "evidence": f"Error executing tool: {e}"
                    })
            else:
                results.append({
                    "plan": plan,
                    "step": step_id,
                    "tool": tool_name,
                    "evidence": f"Tool '{tool_name}' not found."
                })
        return {"results": results}

    def solve(self, state: dict) -> dict:
        """
        Generates the solution based on the evidence.
        This method now returns a streaming-capable chain.
        """
        state["node_history"].append("solve")
        print(f"[solve] Current node history: {state['node_history']}")
        
        prompt_template = state["prompt_template"]
        try:
            result = self.provider.invoke(prompt_template)
            print(result)
            return {"result": result}
        except Exception as e:
            print(e)
            raise e
