https://www.youtube.com/watch?v=1y2TohQdNbo&t=225s


## Setup

```
ollama pull llama3.2:3b

pip install ollama chromadb sentence-transformers streamlit pymupdf langchain-community

ollama pull nomic-embed-text
```

first success

![](2025-01-02-22-11-41.png)

Chroma db created

Can I succesd 3 mb file

3 mb file is not instantaneous

takes more than 5 min

sada

![](2025-01-03-23-38-44.png)

finally workedd

![](2025-01-04-06-20-28.png)

when asked about something irrelevant

![](2025-01-04-07-37-50.png)

```
                                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\aonno\.conda\envs\privategpt\Lib\site-packages\chromadb\api\types.py", line 61, in normalize_embeddings
    raise ValueError(
ValueError: Expected Embedings to be non-empty list or numpy array, got [] in upsert.
```

This could be a result of database corruption

this was resolved by pulling the model again from the repository

ollama pull llama3.2:3b

pip install ollama chromadb sentence-transformers streamlit pymupdf langchain-community

ollama pull nomic-embed-text

Final

![](2025-01-08-23-04-29.png)

