from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3.5:2b", base_url="http://127.0.0.1:11434")
print(repr(llm.invoke("Hi!").content))
