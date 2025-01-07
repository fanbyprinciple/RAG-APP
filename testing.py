import requests

def get_embedding(text, model="nomic-embed-text:latest"):
    url = "http://localhost:11434/api/embeddings"
    payload = {"text": text, "model": model}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()  # Assuming the API returns a list of numbers
    else:
        raise ValueError(f"Error from embedding server: {response.text}")

# Test embedding generation
embedding = get_embedding("This AI is made by Lt Cdr Ashwin")
print(embedding)