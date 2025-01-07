import json
import operator
from typing import TypedDict, List, Annotated
import re

from langchain.chains.summarize.refine_prompts import prompt_template
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages

from pydantic import BaseModel, Field


class ReWOO(TypedDict):
    prompt_template: Annotated[List, add_messages]
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str
    node_history: Annotated[list, operator.add]


class PlanFormatter(BaseModel):
    plan: str = Field(description=f"""plans that are analyzed and broken down from a
                                    task and can solve the problem step by step""")
    tool: str = Field(description=f"""Actions of the plans,
     example: Google[input]: Worker that searches results from Google. Useful when you need to find short
    and succinct answers about a specific topic. The input should be a search query.
    """)
    step: str = Field(description="the sequence number of current step, example: Step#1, Step#2")

class PlannerWorkflow:
    def __init__(self):
        self.graph = StateGraph(ReWOO)
        self.model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
        self.search = TavilySearchResults()
        self.max_iterations = 2  # Prevent infinite loops
        
    def get_plan(self, state: ReWOO) -> dict:
        # Initialize node history if not present
        if "node_history" not in state:
            state["node_history"] = []

        state["node_history"].append("plan")
        print(f"[get_plan] Current node history: {state['node_history']}")
        if self._count_node_visits(state["node_history"], "plan") >= self.max_iterations:
            print(f"Max iterations ({self.max_iterations}) reached. Terminating workflow.")
            return END

        
        
        print("running planner workflow")
        print("state: ")
        print(state)

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
            "tool": "Tool[input]",
            "step": "Step#N"
        }}
        
        Note: You can only call Google tool once in the plan.

        Task: {task}"""

        prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
        planner = prompt_template | self.model
        result = planner.invoke({"task": task})
        
        # Parse the JSON response into PlanFormatter objects
        steps = []
        try:
            # Assuming the response is a list of JSON objects
            plan_steps = json.loads(result.content.replace("```json", "").replace("```", "").strip())
            for step in plan_steps:
                formatted_step = PlanFormatter(**step)
                # Convert to tuple format expected by the rest of the code
                steps.append((
                    formatted_step.plan,
                    formatted_step.step,
                    formatted_step.tool.split('[')[0],  # Extract tool name
                    formatted_step.tool.split('[')[1].rstrip(']')  # Extract input
                ))
        except Exception as e:
            print(f"Error parsing plan: {e}")
            steps = []

        return {"steps": steps, "plan_string": result.content}

    def get_current_task(self, state: ReWOO):
        if "results" not in state or state["results"] is None:
            return 1
        if len(state["results"]) == len(state["steps"]):
            return None
        else:
            return len(state["results"]) + 1

    def tool_execution(self, state: ReWOO) -> dict:
        state["node_history"].append("tool")
        print(f"[tool_execution] Current node history: {state['node_history']}")
        """Worker node that executes the tools of a given plan."""
        _step = self.get_current_task(state)
        _results = None
        # print("steps ", state["steps"])
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
        return {"results": _results}

    def solve(self, state: ReWOO) -> dict:
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
            Response:"""

        plan = ""
        for _plan, step_name, tool, tool_input in state["steps"]:
            _results = state.get("results", {})
            for k, v in _results.items():
                tool_input = tool_input.replace(k, v)
                step_name = step_name.replace(k, v)
            plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
        prompt = solve_prompt.format(plan=plan, task=state["task"])
        result = self.model.invoke(prompt)
        print("Planning execution ended")

        # Return with END marker to ensure termination
        return { "result": [result], END: True }

    def route(self, state: ReWOO) -> str:
        """
        Enhanced routing logic with strict cycle detection and termination after solve
        """
        print(f"[route] Current node history: {state['node_history']}")

        # If 'solve' is in history, terminate the workflow
        if "solve" in state["node_history"]:
            print("Solve has been executed. Terminating workflow.")
            return END  # Directly return END to terminate

        # Check for cycles or max iterations
        if self._has_cycle(state["node_history"]):
            print("Cycle detected in node history. Terminating workflow.")
            return END
        if self._count_node_visits(state["node_history"], "plan") >= self.max_iterations:
            print(f"Max iterations ({self.max_iterations}) reached. Terminating workflow.")
            return END

        _step = self.get_current_task(state)
        if _step is None:
            print("No more tasks to execute. Terminating workflow.")
            return "solve"
        else:
            print(f"Continuing to tool execution. Step: {_step}")
            return "tool"
    
    def _has_cycle(self, history: List[str]) -> bool:
        """
        Detect if there's a cycle in the node traversal history
        """
        pattern = ["plan", "tool", "solve", "plan"]
        str_history = "->".join(history)
        cycle_exists = "->".join(pattern) in str_history
        if cycle_exists:
            print(f"Cycle detected with pattern: {'->'.join(pattern)} in history.")
        return cycle_exists

    def _count_node_visits(self, history: List[str], node: str) -> int:
        """
        Count how many times a specific node has been visited
        """
        count = history.count(node)
        print(f"[_count_node_visits] Node '{node}' visited {count} times.")
        return count

    def compile_graph(self):
        graph = StateGraph(ReWOO)
        graph.add_node("plan", self.get_plan)
        graph.add_node("tool", self.tool_execution)
        graph.add_node("solve", self.solve)
        graph.add_edge("plan", "tool")
        graph.add_edge("solve", END)
        graph.add_conditional_edges("tool", self.route)
        graph.add_edge(START, "plan")

        app = graph.compile()

        return app

    async def stream_token(self, task):
        app = self.compile_graph()

        async for s in app.astream({"task": task}, stream_mode=["messages", "updates"],
                                   config={"recursion_limit": 20}):
            # print(s)
            if s[1][1]["langgraph_node"] == 'solve':

            # print("---")
                yield s