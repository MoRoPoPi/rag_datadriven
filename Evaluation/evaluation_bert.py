# Evaluation code for tha approach using Bert + cosine similarity
# Took 2h using CPU -> results in presentation + repport

import os
import pandas as pd
import torch
import torch.nn.functional as F
import chromadb
import ollama
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# Configuration
CSV_PATH = "resume_job_dataset_updated.csv"
PERSIST_DIR = "./chroma_job_resume_index_bert_100"
COLLECTION_NAME = "evaluation_job_offers_bert_100"
OLLAMA_MODEL = "llama3.2"
MAX_TOKENS = 512

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

class JobMatchingSystem:
    def __init__(self, csv_path, persist_dir, collection_name):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        os.makedirs(self.persist_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)

        existing_collections = [col.name for col in self.chroma_client.list_collections()]
        if collection_name in existing_collections:
            print(f"Using existing Chroma collection: {collection_name}")
            self.collection = self.chroma_client.get_collection(collection_name)
            self.collection_initialized = True
        else:
            print(f"Creating new Chroma collection: {collection_name}")
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.collection_initialized = False

    def normalize_embedding(self, embedding):
        if isinstance(embedding, list):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        return F.normalize(embedding, p=2, dim=0).tolist()

    def get_embedding(self, text):
        try:
            inputs = tokenizer(text, return_tensors="pt", max_length=MAX_TOKENS, truncation=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
            return self.normalize_embedding(embedding)
        except Exception as e:
            print(f"Error embedding text: {e}")
            return self.normalize_embedding(torch.randn(768))

    def summarize_text(self, text):
        if not text or len(text.strip()) < 100:
            return text
        prompt = f"""Summarize the following text concisely, focusing on key skills, experience, and qualifications. Limit the result to fit within 512 tokens:\n{text}"""
        try:
            response = ollama.chat(model=OLLAMA_MODEL, messages=[
                {"role": "system", "content": "You are a professional resume summarizer."},
                {"role": "user", "content": prompt}
            ])
            summary = response['message']['content']
            # Truncate summary to 512 tokens
            tokens = tokenizer.tokenize(summary)[:MAX_TOKENS]
            return tokenizer.convert_tokens_to_string(tokens)
        except Exception as e:
            print(f"Summarization error: {e}")
            return text

    def load_and_process_jobs(self):
        df = pd.read_csv(self.csv_path).fillna("")
        df_jobs = df.drop_duplicates(subset=["job_id"])
        print(f"Loaded {len(df_jobs)} distinct job postings.")

        if not self.collection_initialized:
            job_ids = []
            job_texts = []
            job_embeddings = []
            job_metadata = []

            for index, row in tqdm(df_jobs.iterrows(), total=len(df_jobs), desc="Processing jobs"):
                raw_text = f"Job Title: {row['job_title']}\nDescription: {row['job_description']}"
                summarized = self.summarize_text(raw_text)
                metadata = {col: str(row[col]) for col in df_jobs.columns if pd.notna(row[col])}
                embedding = self.get_embedding(summarized)

                job_ids.append(str(index))
                job_texts.append(summarized)
                job_embeddings.append(embedding)
                job_metadata.append(metadata)

            self.collection.add(
                ids=job_ids,
                documents=job_texts,
                embeddings=job_embeddings,
                metadatas=job_metadata
            )
            print(f"Indexed {len(job_ids)} job postings.")
        else:
            print("Skipping job indexing; collection already exists.")

        return df, df_jobs

    def process_resume(self, resume_text):
        summarized_resume = self.summarize_text(resume_text)
        resume_embedding = self.get_embedding(summarized_resume)
        return summarized_resume, resume_embedding

    def query_jobs(self, resume_text, top_k=5):
        summarized_resume, resume_embedding = self.process_resume(resume_text)
        results = self.collection.query(
            query_embeddings=[resume_embedding],
            n_results=top_k
        )
        matches = []
        if results and 'distances' in results and len(results['distances']) > 0:
            for i, dist in enumerate(results['distances'][0]):
                similarity_score = 1.0 - dist
                matches.append({
                    "id": results['ids'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "text": results['documents'][0][i],
                    "score": similarity_score
                })
        return {
            "original_resume": resume_text,
            "summarized_resume": summarized_resume,
            "matches": matches
        }

    def evaluate_system(self, df_full, k_values=[1, 3, 5, 10, 20], sample_size=50):
        print("Starting comprehensive evaluation...")

        valid_resume_ids = []
        resumes_with_no_related_jobs = 0
        unique_resumes = df_full.drop_duplicates(subset=["resume_id"])[["resume_id", "resume_text", "resume_category"]]
        total_resumes = len(unique_resumes)

        print("Identifying resumes with related jobs...")
        for resume_id in tqdm(unique_resumes["resume_id"], desc="Filtering resumes"):
            related_jobs_df = df_full[(df_full["resume_id"] == resume_id) & (df_full["is_related"] == 1)]
            if not related_jobs_df.empty:
                valid_resume_ids.append(resume_id)
            else:
                resumes_with_no_related_jobs += 1

        print(f"Found {len(valid_resume_ids)} resumes with at least one related job")
        print(f"Excluded {resumes_with_no_related_jobs} resumes with no related jobs")
        print(f"Total resumes in dataset: {total_resumes}")

        if not valid_resume_ids:
            print("ERROR: No valid resumes found with related jobs. Cannot proceed with evaluation.")
            return {k: {"success_rate": 0, "correct_matches": 0, "total_evaluated": 0} for k in k_values}

        jobs_with_related_resumes = df_full[df_full["is_related"] == 1]["job_id"].astype(str).unique()
        df_valid_resumes = unique_resumes[unique_resumes["resume_id"].isin(valid_resume_ids)]

        if sample_size and sample_size < len(df_valid_resumes):
            df_eval_resumes = df_valid_resumes.sample(n=sample_size, random_state=42)
            print(f"Randomly sampled {sample_size} resumes for evaluation")
        else:
            df_eval_resumes = df_valid_resumes
            print(f"Using all {len(df_valid_resumes)} valid resumes for evaluation")

        encountered_unrelated_jobs = set()
        results = {k: {"correct_matches": 0, "total_evaluated": 0} for k in k_values}
        max_k = max(k_values)

        for _, resume_row in tqdm(df_eval_resumes.iterrows(), total=len(df_eval_resumes), desc="Evaluating resumes"):
            resume_id = resume_row["resume_id"]
            resume_text = resume_row["resume_text"]
            resume_category = str(resume_row["resume_category"]).lower().strip()
            related_jobs_df = df_full[(df_full["resume_id"] == resume_id) & (df_full["is_related"] == 1)]
            relevant_job_ids = related_jobs_df["job_id"].astype(str).tolist()

            query_result = self.query_jobs(resume_text, top_k=50)
            all_matches = query_result["matches"]

            filtered_matches = []
            for match in all_matches:
                job_id = str(match["metadata"].get("job_id", match["id"]))
                if job_id in jobs_with_related_resumes:
                    filtered_matches.append(match)
                else:
                    encountered_unrelated_jobs.add(job_id)
            filtered_matches = filtered_matches[:max_k]

            for k in k_values:
                top_k_matches = filtered_matches[:k]
                match_ids = [str(match["metadata"].get("job_id", match["id"])) for match in top_k_matches]
                correct = any(job_id in relevant_job_ids for job_id in match_ids)
                if correct:
                    results[k]["correct_matches"] += 1
                results[k]["total_evaluated"] += 1

        for k in k_values:
            total = results[k]["total_evaluated"]
            correct = results[k]["correct_matches"]
            success_rate = correct / total if total else 0
            results[k]["success_rate"] = success_rate

        return results

jms = JobMatchingSystem(CSV_PATH, PERSIST_DIR, COLLECTION_NAME)
df_full, df_jobs = jms.load_and_process_jobs()
results = jms.evaluate_system(df_full)
print(results)
