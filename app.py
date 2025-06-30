import json
import os

from aci import ACI
from aci.meta_functions import ACISearchFunctions
from aci.types.functions import FunctionDefinitionFormat
from dotenv import load_dotenv
from openai import OpenAI
from rich import print as rprint
from rich.panel import Panel

load_dotenv()
LINKED_ACCOUNT_OWNER_ID = os.getenv("LINKED_ACCOUNT_OWNER_ID", "")
if not LINKED_ACCOUNT_OWNER_ID:
    raise ValueError("LINKED_ACCOUNT_OWNER_ID is not set")

openai = OpenAI()
aci = ACI()

prompt = (
    "You are an expert SRE (Site Reliability Engineering) assistant with access to unlimited tools via ACI_SEARCH_FUNCTIONS. "
    "You can use ACI_SEARCH_FUNCTIONS to find relevant functions across all apps. "
    "Once you have identified the functions you need to use, you can append them to the tools list and use them in future tool calls. "
    
    "You can discover and use functions for ANY task: "
    "- GitHub operations (repositories, issues, commits, content, PRs) "
    "- Cloud services and infrastructure management "
    "- Configuration analysis and deployment reviews "
    "- Monitoring, logging, and observability "
    "- Security analysis and compliance checks "
    "- Cost optimization and resource management "
    "- CI/CD pipeline management "
    "- Database operations "
    "- Web search and research "
    "- And much more... "
    
    "IMPORTANT DEFAULTS: "
    "- For GitHub operations without specified repository, default to 'ehewes/TechEurope' "
    "- For create issue requests, extract title and description from user message "
    "- For list issues requests, format results clearly with issue details "
    "- For configuration analysis, provide detailed security and best practice recommendations "
    
    "Always provide actionable, expert-level SRE recommendations with clear explanations."
)

# ACI meta functions for the LLM to discover the available executable functions dynamically
tools_meta = [
    ACISearchFunctions.to_json_schema(FunctionDefinitionFormat.OPENAI),
]
# store retrieved function definitions (via meta functions) that will be used in the next iteration,
# can dynamically append or remove functions from this list
tools_retrieved: list[dict] = []


def main() -> None:
    # Start the LLM processing loop
    chat_history: list[dict] = []

    while True:
        rprint(Panel("Waiting for LLM Output", style="bold blue"))
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": "Can you review the current PRs for Keploy and review the code and explain it?",
                },
            ]
            + chat_history,
            tools=tools_meta + tools_retrieved,
            parallel_tool_calls=False,
        )

        # Process LLM response and potential function call (there can only be at most one function call)
        content = response.choices[0].message.content
        tool_call = (
            response.choices[0].message.tool_calls[0]
            if response.choices[0].message.tool_calls
            else None
        )
        if content:
            rprint(Panel("LLM Message", style="bold green"))
            rprint(content)
            chat_history.append({"role": "assistant", "content": content})

        if tool_call:
            rprint(
                Panel(f"Function Call: {tool_call.function.name}", style="bold yellow")
            )
            rprint(f"arguments: {tool_call.function.arguments}")

            chat_history.append({"role": "assistant", "tool_calls": [tool_call]})
            result = aci.handle_function_call(
                tool_call.function.name,
                json.loads(tool_call.function.arguments),
                linked_account_owner_id=LINKED_ACCOUNT_OWNER_ID,
                allowed_apps_only=True,
                format=FunctionDefinitionFormat.OPENAI,
            )
            if tool_call.function.name == ACISearchFunctions.get_name():
                tools_retrieved.extend(result)

            rprint(Panel("Function Call Result", style="bold magenta"))
            rprint(result)
            chat_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )
        else:
            rprint(Panel("Task Completed", style="bold green"))
            print("Hello World")
            break


if __name__ == "__main__":
    main()