from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""] 
)

base_dir = Path.cwd()
doc = base_dir / 'data' / 'knowledge.md'

with open(doc,'r') as file:
    all_lines = file.readlines()
    for line in all_lines:
        chunks = text_splitter.split_text(line.strip())
        print(chunks)

