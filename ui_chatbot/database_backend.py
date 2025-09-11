# import dependancies
from langgraph.graph import StateGraph, START, END  # Graph dependancy
from typing import TypedDict, Annotated  # Typing dependancies
from langchain_core.messages import BaseMessage  # Message dependancies
from langchain_ollama import ChatOllama  # LLM dependancies
# from langgraph.checkpoint.memory import InMemorySaver  # Saver dependancies
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages  # Operator in langgraph
import sqlite3


# Create Model
llm = ChatOllama(model="llama3.2:1b")


# Create state schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define chatnode
def chatnode(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect(database='chatbod.db', check_same_thread=False)

# Checkpointer
checkpointer = SqliteSaver(conn = conn)


# Create Graph
graph = StateGraph(ChatState)

# Create node
graph.add_node("chatnode", chatnode)

# Create Graph
graph.add_edge(START, "chatnode")
graph.add_edge("chatnode", END)

# compile graph with checkpointer
chatbot = graph.compile(checkpointer=checkpointer)

# Provide a config argument as required by the list() method
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)