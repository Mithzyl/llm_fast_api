from typing import TypedDict, List
import re

from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from llm.llm_provider import OpenAIProvider


class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str

class PlannerWorkflow:
    def __init__(self):
        # self.llm_provider = OpenAIProvider(base_url="", api_key="")
        self.graph = StateGraph(ReWOO)
        self.model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
        self.search = TavilySearchResults()

    def get_plan(self, state: ReWOO):
        prompt = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
        which external tool together with tool input to retrieve evidence. You can store the evidence into a \
        variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

        Tools can be one of the following:
        (1) Google[input]: Worker that searches results from Google. Useful when you need to find short
        and succinct answers about a specific topic. The input should be a search query.
        (2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
        world knowledge and common sense. Prioritize it when you are confident in solving the problem
        yourself. Input can be any instruction.



        Begin! 
        Describe your plans with rich details. Each Plan should be followed by only one #E.

        Task: {task}"""

        regex_pattern = r"\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
        planner = prompt_template | self.model

        task = state["task"]
        result = planner.invoke({"task": task})
        # Find all matches in the sample text
        matches = re.findall(regex_pattern, result.content)
        return {"steps": matches, "plan_string": result.content}

    def _get_current_task(self, state: ReWOO):
        if "results" not in state or state["results"] is None:
            return 1
        if len(state["results"]) == len(state["steps"]):
            return None
        else:
            return len(state["results"]) + 1

    def tool_execution(self, state: ReWOO):
        """Worker node that executes the tools of a given plan."""
        _step = self._get_current_task(state)
        _results = None
        print("step ", _step)
        # print("steps: ", state["steps"])

        if len(state["steps"]) > 1:
            _, step_name, tool, tool_input = state["steps"][_step - 1]
            _results = (state["results"] or {}) if "results" in state else {}
            for k, v in _results.items():
                tool_input = tool_input.replace(k, v)
            if tool == "Google":
                result = self.search.invoke(tool_input)
            elif tool == "LLM":
                result = self.model.invoke(tool_input)
            else:
                raise ValueError
            _results[step_name] = str(result)
        return {"results": _results if _results else None}

    def solve(self, state: ReWOO):
        solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
        retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
        contain irrelevant information.

        {plan}

        Now solve the question or task according to provided Evidence above. Respond with the answer
        directly with no extra words.

        Task: {task}
        Response:"""

        plan = ""
        for _plan, step_name, tool, tool_input in state["steps"]:
            _results = (state["results"] or {}) if "results" in state else {}
            for k, v in _results.items():
                tool_input = tool_input.replace(k, v)
                step_name = step_name.replace(k, v)
            plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
        prompt = solve_prompt.format(plan=plan, task=state["task"])
        result = self.model.invoke(prompt)
        return {"result": result.content}

    def _route(self, state):
        _step = self._get_current_task(state)
        if _step is None:
            # We have executed all tasks
            return "solve"
        else:
            # We are still executing tasks, loop back to the "tool" node
            return "tool"

    def compile_graph(self):
        graph = StateGraph(ReWOO)
        graph.add_node("plan", self.get_plan)
        graph.add_node("tool", self.tool_execution)
        graph.add_node("solve", self.solve)
        graph.add_edge("plan", "tool")
        graph.add_edge("solve", END)
        graph.add_conditional_edges("tool", self._route)
        graph.add_edge(START, "plan")

        app = graph.compile()

        return app

    async def stream_token(self, task):
        app = self.compile_graph()

        async for s in app.astream({"task": task}, stream_mode=["messages"]):
            # print(s)
            # print("---")
            yield s