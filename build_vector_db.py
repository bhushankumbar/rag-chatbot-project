import sqlite3
import pandas as pd
import chromadb
import shutil
import os
from chromadb.utils import embedding_functions

# 1. Clean up old database folder safely so files don't lock up
if os.path.exists("./chroma_db"):
    try:
        shutil.rmtree("./chroma_db")
    except Exception:
        pass

# 2. Pull recent titles from 2019 onwards (optimized for speed!)
conn = sqlite3.connect("netflix.db")
query = "SELECT title, description FROM clean_titles WHERE release_year >= 2019"
df = pd.read_sql_query(query, conn)
conn.close()

# Drop any empty rows just in case
df = df.dropna(subset=['title', 'description'])

print(f"🚀 Found {len(df)} recent movies (2019+). Starting fast batch vectorization...")

# 3. Setup Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)
collection = chroma_client.get_or_create_collection(name="netflix_collection", embedding_function=ollama_ef)

# 4. Fast Batch Processing (100 movies at a time!)
batch_size = 100
for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i:i+batch_size]
    
    documents_batch = [f"Title: {row['title']}. Description: {row['description']}" for _, row in batch_df.iterrows()]
    ids_batch = [str(idx) for idx in batch_df.index]
    
    collection.add(documents=documents_batch, ids=ids_batch)
    print(f"📦 Progress: Indexed {min(i + batch_size, len(df))}/{len(df)} movies...")

print("✅ Success: Your modern Netflix dataset is loaded and vectorized!")
