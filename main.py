import chromadb
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="./chroma_db"
    )
)

collection = client.get_or_create_collection(name="knowledge")

doc = Path("data/knowledge.md")
full_text = doc.read_text()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = text_splitter.split_text(full_text)

embeddings = model.encode(chunks, convert_to_numpy=True)

print(f"Chunks: {len(chunks)}")
print(f"Embedding shape: {embeddings.shape}")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(chunks))]
)

print("✅ Stored in ChromaDB")
