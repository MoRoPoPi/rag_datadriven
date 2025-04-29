import os
import chromadb
import pandas as pd
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

CSV_PATH = "postings.csv"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "postings"
EMBED_MODEL_NAME = "nomic-embed-text"

def load_documents_from_csv(filepath):
    """Loads data from CSV file into documents."""
    print(f"Loading documents from {filepath}...")
    try:
        df = pd.read_csv(filepath).fillna("")
        
        documents = []
        for index, row in df.iterrows():
            text_parts = []
            doc_metadata = {'csv_row_index': index}
            
            # Add text parts and metadata
            if 'title' in df.columns and row['title']:
                text_parts.append(f"Job Title: {row['title']}")
                doc_metadata['title'] = row['title']
            
            if 'company_name' in df.columns and row['company_name']:
                text_parts.append(f"Company: {row['company_name']}")
                doc_metadata['company_name'] = row['company_name']

            if 'description' in df.columns and row['description']:
                text_parts.append(f"Description: {row['description']}")
                # Consider chunking description if very long

            if 'skills_desc' in df.columns and row['skills_desc']:
                text_parts.append(f"Skills: {row['skills_desc']}")
                # Potentially extract key skills into metadata list

            # Add other relevant fields as metadata
            if 'location' in df.columns and row['location']:
                doc_metadata['location'] = row['location']
            if 'min_salary' in df.columns and row['min_salary']:
                doc_metadata['min_salary'] = float(row['min_salary']) if row['min_salary'] else 0.0
            if 'max_salary' in df.columns and row['max_salary']:
                doc_metadata['max_salary'] = float(row['max_salary']) if row['max_salary'] else 0.0
            if 'formatted_work_type' in df.columns and row['formatted_work_type']:
                doc_metadata['work_type'] = row['formatted_work_type']
            if 'formatted_experience_level' in df.columns and row['formatted_experience_level']:
                doc_metadata['experience_level'] = row['formatted_experience_level']
            if 'remote_allowed' in df.columns:
                 doc_metadata['remote_allowed'] = bool(row['remote_allowed']) if pd.notna(row['remote_allowed']) else None


            doc_text = "\n\n".join(text_parts)
            
            documents.append(Document(text=doc_text, metadata=doc_metadata))
        
        print(f"Loaded {len(documents)} documents with enriched metadata.")
        return documents
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []


def build_index():
    """Builds and persists a simple vector index."""
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)
    
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
    
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    
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
