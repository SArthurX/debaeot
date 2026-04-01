import time
import requests
import json
from langchain_ollama import ChatOllama

def test_raw_api():
    print("Testing Raw python requests to Ollama...")
    start = time.time()
    resp = requests.post("http://127.0.0.1:11434/api/chat", json={
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": "你好，請寫一篇100字的小短文。"}],
        "stream": True
    }, stream=True)
    
    first_token_time = None
    for line in resp.iter_lines():
        if line:
            if not first_token_time:
                first_token_time = time.time()
            # break early just to test TTFT
            break
            
    print(f"Raw API Time to First Token (TTFT): {first_token_time - start:.4f} sec")


def test_langchain():
    print("Testing LangChain ChatOllama...")
    start = time.time()
    llm = ChatOllama(model="qwen3.5:9b", base_url="http://127.0.0.1:11434", temperature=0.7)
    
    first_token_time = None
    for chunk in llm.stream("你好，請寫一篇100字的小短文。"):
        if not first_token_time:
            first_token_time = time.time()
            break
            
    print(f"LangChain TTFT: {first_token_time - start:.4f} sec")

if __name__ == "__main__":
    test_raw_api()
    test_langchain()
