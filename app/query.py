import os
import argparse
import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "postings"
EMBED_MODEL_NAME = "nomic-embed-text"

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

class NormalizedOllamaEmbedding(OllamaEmbedding):
    def get_text_embedding(self, text):
        raw_vec = super().get_text_embedding(text)
        return normalize(np.array(raw_vec)).tolist()

def query_index(query_text, return_nodes=False, similarity_top_k=5):
    embed_model = NormalizedOllamaEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    if not os.path.exists(PERSIST_DIR):
        print(f"Error: Index directory not found at {PERSIST_DIR}")
        return None

    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=PERSIST_DIR
    )
    index = load_index_from_storage(storage_context)

    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    retrieved_nodes = retriever.retrieve(query_text)

    query_embedding = normalize(np.array(embed_model.get_text_embedding(query_text)))

    scored_nodes = []
    for i, node in enumerate(retrieved_nodes):
        doc_text = node.node.get_content()
        doc_embedding = normalize(np.array(embed_model.get_text_embedding(doc_text)))
        score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        node.score = score
        scored_nodes.append(node)

    # Sort nodes manually by cosine similarity
    scored_nodes.sort(key=lambda n: n.score, reverse=True)

    print("\n==== Retrieved Documents ====")
    for node in scored_nodes:
        print(f"\n---(Score: {node.score:.4f}) ---")
        print(node.node.get_content()[:300]) 

    if return_nodes:
        return scored_nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for relevant job postings.")
    parser.add_argument("query", type=str, help="The search query")
    args = parser.parse_args()

    query_index(args.query)
