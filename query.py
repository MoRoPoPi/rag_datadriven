import os
import chromadb
import argparse
from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

# Simple configuration
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "postings"
EMBED_MODEL_NAME = "nomic-embed-text"

def query_index(query_text, return_nodes=False, similarity_top_k=20):
    """Simple function to query the index and return relevant documents."""
    # Configure embeddings
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)
    
    # Check if index exists
    if not os.path.exists(PERSIST_DIR):
        print(f"Error: Index directory not found at {PERSIST_DIR}")
        return None
    
    # Initialize ChromaDB
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_collection(COLLECTION_NAME)
    
    # Create vector store and load index
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=PERSIST_DIR
    )
    
    # Load the index
    index = load_index_from_storage(storage_context)
    
    # Create retriever and retrieve documents
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    retrieved_nodes = retriever.retrieve(query_text)
    
    # Return the nodes or display them
    if return_nodes:
        return retrieved_nodes
    else:
        print("\n==== Retrieved Documents ====")
        for i, node in enumerate(retrieved_nodes):
            print(f"\n--- Document {i+1} (Score: {node.score:.4f}) ---")
            print(node.node.get_content())
    
    return retrieved_nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for relevant job postings.")
    parser.add_argument("query", type=str, help="The search query")
    args = parser.parse_args()
    
    query_index(args.query)
