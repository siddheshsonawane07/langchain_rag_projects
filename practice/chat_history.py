from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage

local_llm = "llama3.2"
model = ChatOllama(model=local_llm, temperature=0)

chat_history = [] 

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)  

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query)) 

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response)) 

    print(f"AI: {response}")


print("---- Message History ----")
print(chat_history)