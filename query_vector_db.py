import chromadb
from chromadb.utils import embedding_functions

# 1. Connect to your local Vector Database folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. Use the exact same math engine to convert your questions into vectors
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

# 3. Load your stored Netflix collection
collection = chroma_client.get_collection(name="netflix_collection", embedding_function=ollama_ef)

# 4. Start the interactive chat loop
print("\n🎬 Netflix AI Search Bot is online! Type 'exit' to quit.")
while True:
    user_query = input("\n🧐 What kind of movie/show are you looking for? ")
    
    if user_query.lower() == 'exit':
        print("Goodbye! 👋")
        break
        
    if not user_query.strip():
        continue

    # 5. Search the vector database for the top 3 best matching items
    results = collection.query(
        query_texts=[user_query],
        n_results=3
    )

    # 6. Print out the results neatly
    print("\n🍿 Top 3 Matches Found:")
    for doc in results['documents'][0]:
        print(f"👉 {doc}")
