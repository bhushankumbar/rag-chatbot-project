import streamlit as st
import chromadb
import ollama
from chromadb.utils import embedding_functions

# Set up webpage layout
st.set_page_config(page_title="Netflix AI Search", page_icon="🍿", layout="wide")
st.title("🎬 Netflix AI Movie Chatbot")
st.write("Tell the AI chatbot your mood, and it will pick the best options from our database and explain why!")

@st.cache_resource
def get_vector_db():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    )
    return chroma_client.get_collection(name="netflix_collection", embedding_function=ollama_ef)

try:
    collection = get_vector_db()
    user_query = st.text_input("🧐 What kind of movie or show are you looking for?", placeholder="e.g., a sci-fi thriller with mind-bending twists")

    if user_query:
        with st.spinner("🧠 Step 1: Searching database vectors..."):
            results = collection.query(
                query_texts=[user_query],
                n_results=5
            )
            
        # FIXED: Correctly extracts individual items out of ChromaDB's nested list package
        raw_docs = results['documents']
        doc_list = raw_docs[0] if (raw_docs and isinstance(raw_docs, list) and isinstance(raw_docs[0], list)) else raw_docs
        
        # Safely convert everything into standard string text for the AI model
        cleaned_docs = [str(d) for d in doc_list if d]
        context_text = "\n\n".join(cleaned_docs)
        
        if context_text.strip():
            with st.spinner("🤖 Step 2: Sending data to Gemma3 for evaluation..."):
                prompt = f"""
                You are a helpful and passionate Netflix cinema recommendation expert. 
                The user is looking for a movie with this mood or style: "{user_query}".
                
                Here are the top 5 best matching movies found in our database catalog:
                {context_text}
                
                Please read these matches carefully and write a warm, engaging response recommending these films to the user. 
                Explain briefly why each film fits the user's request. Keep your tone enthusiastic and professional.
                """
                
                response = ollama.generate(model="gemma3", prompt=prompt)
                ai_response_text = response['response']
                
            st.write("---")
            st.subheader("🤖 Your AI Assistant's Recommendations:")
            st.write(ai_response_text)
        else:
            st.warning("No matching movies found in the database for this query.")

except Exception as e:
    st.error(f"Could not complete the request. Error: {e}")
