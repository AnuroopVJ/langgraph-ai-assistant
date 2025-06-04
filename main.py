from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from duckduckgo_search import DDGS
from typing_extensions import Literal
import subprocess
import os

llm = init_chat_model("groq:llama3-8b-8192")
ROOT_DIR = "<path>"

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic"""
    try:
        results = DDGS().text(query, max_results=3)
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(f"{i}. {result.get('title', 'No title')}\n   {result.get('body', 'No description')}\n   URL: {result.get('href', 'No URL')}")
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def create_file(filename: str, content: str) -> str:
    """Create a new file with the specified content"""
    try:
        base_dir = ROOT_DIR
        full_path = os.path.join(base_dir, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully created file '{filename}' with content"
    except Exception as e:
        return f"Error creating file: {str(e)}"

@tool
def read_file(filename: str) -> str:
    """Read the contents of a file"""
    try:
        base_dir = ROOT_DIR
        full_path = os.path.join(base_dir, filename)
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"Contents of '{filename}':\n{content}"
        else:
            return f"File '{filename}' does not exist"
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_to_file(filename: str, content: str) -> str:
    """Write content to an existing file (overwrites existing content)"""
    try:
        base_dir = r"C:\Users\neela\Desktop\Miscellaneous\coding\All_in_one_Agent\ai_dir"
        full_path = os.path.join(base_dir, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote content to '{filename}'"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

@tool
def list_files() -> str:
    """List all files in the working directory"""
    try:
        base_dir = r"C:\Users\neela\Desktop\Miscellaneous\coding\All_in_one_Agent\ai_dir"
        if os.path.exists(base_dir):
            files = os.listdir(base_dir)
            if files:
                return f"Files in directory:\n" + "\n".join(f"- {file}" for file in files)
            else:
                return "Directory is empty"
        else:
            return "Directory does not exist"
    except Exception as e:
        return f"Error listing files: {str(e)}"

@tool
def delete_file(filename: str) -> str:
    """Delete a file"""
    try:
        base_dir = r"C:\Users\neela\Desktop\Miscellaneous\coding\All_in_one_Agent\ai_dir"
        full_path = os.path.join(base_dir, filename)
        if os.path.exists(full_path):
            os.remove(full_path)
            return f"Successfully deleted file '{filename}'"
        else:
            return f"File '{filename}' does not exist"
    except Exception as e:
        return f"Error deleting file: {str(e)}"

@tool
def run_command(command: str) -> str:
    """Execute a shell command"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout if result.stdout else result.stderr
        return f"Command: {command}\nOutput:\n{output}"
    except subprocess.TimeoutExpired:
        return f"Command '{command}' timed out after 30 seconds"
    except Exception as e:
        return f"Error running command '{command}': {str(e)}"

@tool
def add_todo_task(task: str) -> str:
    """Add a task to the to-do list"""
    try:
        todo_file = "to_do_list.txt"
        # Create file if it doesn't exist
        if not os.path.exists(todo_file):
            with open(todo_file, "w", encoding='utf-8') as f:
                f.write("TO-DO LIST\n==========\n")
        
        with open(todo_file, "a", encoding='utf-8') as f:
            f.write(f"- {task}\n")
        return f"Added task: '{task}'"
    except Exception as e:
        return f"Error adding task: {str(e)}"

@tool
def view_todo_list() -> str:
    """View the current to-do list"""
    try:
        todo_file = "to_do_list.txt"
        if os.path.exists(todo_file):
            with open(todo_file, "r", encoding='utf-8') as f:
                content = f.read()
            return f"Current to-do list:\n{content}"
        else:
            return "No to-do list found. Use add_todo_task to create one."
    except Exception as e:
        return f"Error viewing to-do list: {str(e)}"

@tool
def remove_todo_task(task_keyword: str) -> str:
    """Remove a task from the to-do list by keyword"""
    try:
        todo_file = "to_do_list.txt"
        if not os.path.exists(todo_file):
            return "No to-do list found"
        
        with open(todo_file, "r", encoding='utf-8') as f:
            lines = f.readlines()
        
        original_count = len(lines)
        updated_lines = [line for line in lines if task_keyword.lower() not in line.lower()]
        
        if len(updated_lines) < original_count:
            with open(todo_file, "w", encoding='utf-8') as f:
                f.writelines(updated_lines)
            return f"Removed task(s) containing '{task_keyword}'"
        else:
            return f"No tasks found containing '{task_keyword}'"
    except Exception as e:
        return f"Error removing task: {str(e)}"

# Define all tools
tools = [
    search_web, 
    create_file, 
    read_file, 
    write_to_file, 
    list_files, 
    delete_file, 
    run_command, 
    add_todo_task, 
    view_todo_list, 
    remove_todo_task
]

llm_with_tools = llm.bind_tools(tools)

def llm_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

tool_node = ToolNode(tools)

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("agent", llm_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()

if __name__ == "__main__":
    # Ensure the ai_dir exists
    ai_dir = r"C:\Users\neela\Desktop\Miscellaneous\coding\All_in_one_Agent\ai_dir"
    os.makedirs(ai_dir, exist_ok=True)
    
    
    print("\n" + "="*50 + "\n")
    
    while True:   
        usr_inp = input(">> ")

        if usr_inp.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break

        try:
            initial_state = {"messages": [{"role": "user", "content": usr_inp}]}
            final_state = graph.invoke(initial_state, config={"recursion_limit": 50})
            
            last_message = final_state["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content:
                print(f"\nAssistant: {last_message.content}\n")
            else:
                print(f"\nAssistant: {last_message}\n")
                
        except Exception as e:
            print(f"❌ Error: {e}\n")