import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "llama3.1:latest",
    "prompt": "Explain AI in one sentence.",
    "stream": False
}

response = requests.post(url, json=payload)

print("RAW:", response.text)
