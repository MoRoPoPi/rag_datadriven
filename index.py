import os
import chromadb
import pandas as pd
import numpy as np
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from chromadb.config import Settings as ChromaSettings

CSV_PATH = "./postings.csv"
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

def load_documents_from_csv(filepath):
    print(f"Loading documents from {filepath}...")
    try:
        df = pd.read_csv(filepath).fillna("")
        documents = []
        
        df = df.head(500) # load the first 500 rows
        for index, row in df.iterrows():
            text_parts = []
            doc_metadata = {'csv_row_index': index}
            
            if 'title' in df.columns and row['title']:
                text_parts.append(f"Job Title is: {row['title']}")
                doc_metadata['title'] = row['title']
            if 'company_name' in df.columns and row['company_name']:
                doc_metadata['company_name'] = row['company_name']
            if 'description' in df.columns and row['description']:
                text_parts.append(f"Description of the job is: {row['description']}")
            if 'skills_desc' in df.columns and row['skills_desc']:
                text_parts.append(f"Skills Needed are: {row['skills_desc']}")
            if 'location' in df.columns and row['location']:
                doc_metadata['location'] = row['location']
            if 'min_salary' in df.columns and row['min_salary']:
                doc_metadata['min_salary'] = float(row['min_salary'])
            if 'max_salary' in df.columns and row['max_salary']:
                doc_metadata['max_salary'] = float(row['max_salary'])
            if 'formatted_work_type' in df.columns and row['formatted_work_type']:
                doc_metadata['work_type'] = row['formatted_work_type']
            if 'formatted_experience_level' in df.columns and row['formatted_experience_level']:
                doc_metadata['experience_level'] = row['formatted_experience_level']
            if 'remote_allowed' in df.columns:
                doc_metadata['remote_allowed'] = bool(row['remote_allowed']) if pd.notna(row['remote_allowed']) else None

            doc_text = "\n".join(text_parts)
            documents.append(Document(text=doc_text, metadata=doc_metadata))
        
        print(f"Loaded {len(documents)} documents with enriched metadata.")
        return documents
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

def build_index():
    Settings.embed_model = NormalizedOllamaEmbedding(model_name=EMBED_MODEL_NAME)
    
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)

    db = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=ChromaSettings(allow_reset=True)
    )
    # db.delete_collection(COLLECTION_NAME)

    chroma_collection = db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    documents = load_documents_from_csv(CSV_PATH)

    if not documents:
        print("No documents to index. Aborting.")
        return
    
    print("Building index...")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index built and saved to {PERSIST_DIR}")

if __name__ == "__main__":
    build_index()
