# import dependancies
from langgraph.graph import StateGraph, START, END # Graph dependancy
from typing import TypedDict, Annotated # Typing dependancies
from langchain_core.messages import BaseMessage # Message dependancies
from langchain_ollama import ChatOllama # LLM dependancies
from langgraph.checkpoint.memory import InMemorySaver # Saver dependancies
from langgraph.graph.message import add_messages # Operator in langgraph


# Create Model
llm = ChatOllama(model= "llama3.2:1b")

# Create state schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define chatnode
def chatnode(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


# Checkpointer
checkpointer = InMemorySaver()


# Create Graph
graph = StateGraph(ChatState)

# Create node
graph.add_node("chatnode", chatnode)

# Create Graph
graph.add_edge(START, 'chatnode')
graph.add_edge('chatnode', END)

# compile graph with checkpointer
chatbot = graph.compile(checkpointer=checkpointer)
