from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

load_dotenv()

def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  
    now = datetime.datetime.now()  
    return now.strftime("%I:%M %p")

# List of tools available to the agent
tools = [
    Tool(
        name="Time",  
        func=get_current_time, 
        description="Useful for when you need to know the current time",
    ),
]

prompt = hub.pull("hwchase17/react")

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create an agent executor with handle_parsing_errors=True
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  
    verbose=True,
    max_iterations=3 
)

try:
    response = agent_executor.invoke({"input": "What time is it?"})
    print("Response:", response)
except Exception as e:
    print(f"An error occurred: {str(e)}")



'''
The ReAct (Reason + Act) agent works by:

First reasoning about the question or input.
Deciding on an action (such as calling the time-fetching tool).
Generating a final response based on the action’s output.

The AgentExecutor is a component in LangChain that manages the execution flow of an agent. It coordinates the agent's reasoning and action-taking processes, ensuring that each step follows the agent’s intended logic and tool usage. Here’s a deeper breakdown of what it does and why it’s helpful:


'''