import json
import os
from typing import List, Dict, Any
import sys

# Add the 'src' directory to the Python path to resolve module imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Set ROOT_DIR environment variable
if "ROOT_DIR" not in os.environ:
    os.environ["ROOT_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from duckduckgo_search import DDGS
from langchain.chains.summarize.refine_prompts import prompt_template
from langsmith import traceable
from mcp.server.fastmcp import FastMCP
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from llm.llm_factory import get_llm_provider

mcp = FastMCP("AgentTools")

llm_provider = get_llm_provider(model="deepseek-chat")

@mcp.tool()
@traceable
def get_plan(task: str, tools_json: str) -> str:
    """
    Generates a step-by-step plan for a given task using an LLM.
    'tools_json' should be a JSON string representing the list of available tools.
    The plan will be returned as a JSON string.
    """
    try:
        tools = json.loads(tools_json)
        tool_prompt_strings = []
        for i, tool in enumerate(tools):
            tool_prompt_strings.append(f"[{i + 1}]: {tool['name']}[input]: {tool['description']}")
        tool_prompt_strings = "\n".join(tool_prompt_strings)
    except (json.JSONDecodeError, KeyError) as e:
        return json.dumps({"error": f"Invalid JSON format for tools: {e}"})


    prompt_template_str = f"""For the following task, make plans that can solve the problem step by step. For each plan, indicate \
    which external tool together with tool input to retrieve evidence.

    Tools can be one of the following:
    {tool_prompt_strings}

    Your response should be in JSON format with the following structure for each step:
    {{{{
        "plan": "Detailed description of the plan",
        "tool": "Tool",
        "step": "Step#N",
        "query": "Query strings"
    }}}}
    Do not wrap the json result in markdown format or json markers
    
    Example(tool in the example may not be available, you should read the tool list provided as proof):
    [
        {{{{
            "plan": "Understand the concept of black holes, their formation, and properties to ensure foundational knowledge.",
            "tool": "search",
            "step": "Step#1",
            "query": "Latest discoveries about black holes in astrophysics."
        }}}}
    ]
    
    Note: You can only select the tools within the tools provided, and the current tool is get plan tool, so you don't
        need to plan this tool again. So does solve node.
        When you think the task is related to complicated problems related to math, coding, and thesis writing, you need
        to think about taking usage of prompt rewriting
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template_str),
        ("human", "{task}")
    ])

    planner = prompt | llm_provider
    result = planner.invoke({"task": task})

    return result.content


@mcp.tool()
@traceable
def search(query: str, max_results: int = 5) -> str:
    """
    Performs a web search using DuckDuckGo.
    'query' is the search query.
    'max_results' is the maximum number of results to return.
    """
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
            return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": f"An error occurred during search: {e}"})


@mcp.tool()
@traceable
async def solve(results_json: str, task: str) -> str:
    """
    Final node in the planner workflow that generates the solution based on the evidence.
    'results_json' should be a JSON string of the results from tool_execution.
    'task' is the original user task.
    """
    try:
        results = json.loads(results_json)
        solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
            retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
            contain irrelevant information.

            {plan}

            Now solve the question or task according to provided Evidence above. Respond with the answer
            according to plan results and respond in markdown format.

            Task: {task}
            """

        plan_summary = ""
        for result in results:
            plan_summary += f"Plan: {result['plan']}\n{result['step']} = {result['tool']}[{result['evidence']}]\n"

        prompt = solve_prompt.format(plan=plan_summary, task=task)
        
        prompt_template = [SystemMessage(prompt)]
        
        result = await llm_provider.ainvoke(prompt_template)
        return result.content
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON format for results: {e}"})
    except Exception as e:
        return json.dumps({"error": f"An error occurred during the solve step: {e}"})


if __name__ == "__main__":
    # This allows the MCP server to be run directly
    print("Loading local MCP agent server")
    mcp.run(transport="stdio")
