import os
import chromadb
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from google import genai

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# client = chromadb.Client(
#     chromadb.config.Settings(
#         persist_directory="./chroma_db"
#     )
# )

# collection = client.get_or_create_collection(name="knowledge")

# doc = Path("data/knowledge.md")
# full_text = doc.read_text()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100
# )

# chunks = text_splitter.split_text(full_text)

# embeddings = model.encode(chunks, convert_to_numpy=True)

# collection.add(
#     documents=chunks,
#     embeddings=embeddings.tolist(),
#     ids=[str(i) for i in range(len(chunks))]
# )

# question = "what are the plans available"
# question_embedding = model.encode(question, convert_to_numpy=True)

# results = collection.query(
#     query_embeddings=[question_embedding],
#     n_results=3
# )

# retrieved_docs = results["documents"][0]

# context = "\n\n".join(retrieved_docs)

# prompt = f"""
# You are a helpful assistant.
# Answer ONLY using the context below.
# If the answer is not present, say "I don't know".

# Context:
# {context}

# Question:
# {question}
# """

# gemini_client = genai.Client(
#     api_key=os.getenv("GEMINI_API_KEY")
# )

# response = gemini_client.models.generate_content(
#     model="gemini-3-flash-preview",
#     contents=prompt
# )

# print(response.text)
