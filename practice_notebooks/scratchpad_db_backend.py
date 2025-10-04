from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3


llm = ChatOllama(model= "llama3.2:1b")

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatnode(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


graph = StateGraph(ChatState)

graph.add_node("chatnode", chatnode)

graph.add_edge(START, 'chatnode')
graph.add_edge('chatnode', END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)



