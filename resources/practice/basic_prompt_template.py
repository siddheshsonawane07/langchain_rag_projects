from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# using template string
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

print("-----Prompt from Template-----")
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)

# PART 2: Prompt with Multiple Placeholders
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
print("\n----- Prompt with Multiple Placeholders -----\n")
print(prompt)


messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)


messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 3 jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers"})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)

