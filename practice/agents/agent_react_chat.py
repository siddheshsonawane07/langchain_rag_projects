from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

load_dotenv()

def get_current_time():
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    try:
        # Limit to two sentences for brevity
        print("je;;p")
    except:
        return "I couldn't find any information on that."