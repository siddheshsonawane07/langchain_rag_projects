from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_ollama import ChatOllama


model = ChatOllama(model="llama3.2")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),  
        ("human", "Tell me {joke_count} jokes."),     ]                 
)

# Create individual runnables (steps in the chain) to process input and produce output
# This step formats the prompt template with specific variables (topic and joke_count)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))

# This step invokes the ChatOpenAI model with the formatted messages
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))

# This step extracts the content from the model's output
parse_output = RunnableLambda(lambda x: x.content)

# Create a RunnableSequence, which is a sequence of tasks that will be executed in order
# The sequence first formats the prompt, then invokes the model, and finally parses the output
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain with specific input: the topic "lawyers" and requesting 3 jokes
response = chain.invoke({"topic": "software engineers", "joke_count": 3})

print(response)
