from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from duckduckgo_search import DDGS
from typing_extensions import Literal
from langchain_community.agent_toolkits import FileManagementToolkit
import subprocess
import os

llm = init_chat_model("groq:llama3-8b-8192")


class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def search_tool(query: str):
    """Search the web for a query (up-to-date information)"""
    results = DDGS().text(query, max_results=2)
    return str(results)

@tool
def file_system_interaction(query: Literal["read", "write", "delete", "list", "create"], file_or_dir: str, content: str = ""):
    """Interact with the file system. 
    - query: The operation to perform (read, write, delete, list, create)
    - file_or_dir: The file or directory path
    - content: Content to write (only used for write and create operations)
    """
    base_dir = r"C:\Users\neela\Desktop\Miscellaneous\coding\All_in_one_Agent\ai_dir"
    full_path = os.path.join(base_dir, file_or_dir)
    
    try:
        if query == "list":
            if os.path.exists(full_path) and os.path.isdir(full_path):
                files = os.listdir(full_path)
                print("LISTED")
                return f"Directory contents: {files}"
            else:
                files = os.listdir(base_dir)
                return f"Base directory contents: {files}"
                
        elif query == "read":
            if os.path.exists(full_path):
                print("READ")
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return f"File content:\n{content}"
            else:
                return f"File {file_or_dir} does not exist"
                
        elif query == "write" or query == "create":
            print("WRITE")
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully {'created' if query == 'create' else 'wrote to'} file {file_or_dir}"
            
        elif query == "delete":
            print("DELETE")
            if os.path.exists(full_path):
                os.remove(full_path)
                return f"Successfully deleted file {file_or_dir}"
            else:
                return f"File {file_or_dir} does not exist"
                
    except Exception as e:
        print(f"Error performing {query} operation: {str(e)}")
        return f"Error performing {query} operation: {str(e)}"

@tool
def shell(cmd: str):
    """Run a shell command"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout if result.stdout else result.stderr
        return f"Command output:\n{output}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {str(e)}"

@tool
def to_do_list(action: Literal["create", "add", "remove", "view"], task: str = "", current_list: str = ""):
    """Manage a to-do list.
    - action: The action to perform (create, add, remove, view)
    - task: The task to add or remove
    - current_list: The current list content (used for remove operations)
    """
    todo_file = "to_do_list.txt"
    
    try:
        if action == "create":
            with open(todo_file, "w", encoding='utf-8') as f:
                f.write(f"TO-DO LIST\n==========\n- {task}\n" if task else "TO-DO LIST\n==========\n")
            return "New to-do list created"
            
        elif action == "add":
            with open(todo_file, "a", encoding='utf-8') as f:
                f.write(f"- {task}\n")
            return f"Task '{task}' added to to-do list"
            
        elif action == "remove":
            if os.path.exists(todo_file):
                with open(todo_file, "r", encoding='utf-8') as f:
                    content = f.read()
                # Remove the task line
                lines = content.split('\n')
                updated_lines = [line for line in lines if task.lower() not in line.lower()]
                with open(todo_file, "w", encoding='utf-8') as f:
                    f.write('\n'.join(updated_lines))
                return f"Task containing '{task}' removed from to-do list"
            else:
                return "To-do list file does not exist"
                
        elif action == "view":
            if os.path.exists(todo_file):
                with open(todo_file, "r", encoding='utf-8') as f:
                    content = f.read()
                return f"Current to-do list:\n{content}"
            else:
                return "To-do list file does not exist"
                
    except Exception as e:
        return f"Error managing to-do list: {str(e)}"

tools = [search_tool, file_system_interaction, shell, to_do_list]

llm_with_tools = llm.bind_tools(tools)

def llm_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def router_node(state: State):
    last_message = state["messages"][-1]
    return 'tools' if getattr(last_message, 'tool_calls', None) else 'llm'

tool_node = ToolNode(tools)

# Build the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("tools", "llm")
graph_builder.add_conditional_edges('llm', router_node, {'tools': 'tools', 'llm': 'llm'})

graph = graph_builder.compile()

if __name__ == "__main__":
    # Ensure the ai_dir exists
    ai_dir = r"C:\Users\neela\Desktop\Miscellaneous\coding\All_in_one_Agent\ai_dir"
    os.makedirs(ai_dir, exist_ok=True)
    

    
    while True:   
        try:
            state = {"messages": []}
            usr_inp = input(">> ")
            if usr_inp.lower() == "exit":
                break
            state["messages"].append({"role": "user", "content": usr_inp})

            # Run the graph
            state = graph.invoke(state)
            # Print the response
            print(state["messages"][-1]["content"])
        except Exception as e:
            print(f"Error: {e}")