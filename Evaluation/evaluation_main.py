# Evaluation code for tha approach using nomic-embed-text + cosine similarity
# Took 2h using CPU -> results in presentation + report

import os
import pandas as pd
import torch
import torch.nn.functional as F
import chromadb
import ollama
from tqdm import tqdm

# Configuration
CSV_PATH = "resume_job_dataset_updated.csv"
PERSIST_DIR = "./chroma_job_resume_index"
COLLECTION_NAME = "evaluation_job_offers"
OLLAMA_MODEL = "llama3.2"
EMBED_MODEL_NAME = "nomic-embed-text"

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
            response = ollama.embeddings(model=EMBED_MODEL_NAME, prompt=text)
            embedding = response.get('embedding', [])
            return self.normalize_embedding(embedding)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return self.normalize_embedding(torch.randn(384))

    def summarize_text(self, text):
        if not text or len(text.strip()) < 100:
            return text
        prompt = f"""Summarize the following text concisely, focusing on key skills, 
                   experience, and qualifications. Limit to 500 words:\n{text}"""
        try:
            response = ollama.chat(model=OLLAMA_MODEL, messages=[
                {"role": "system", "content": "You are a professional resume summarizer."},
                {"role": "user", "content": prompt}
            ])
            return response['message']['content']
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
                text = f"Job Title: {row['job_title']}\nDescription: {row['job_description']}"
                metadata = {col: str(row[col]) for col in df_jobs.columns if pd.notna(row[col])}
                embedding = self.get_embedding(text)

                job_ids.append(str(index))
                job_texts.append(text)
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

    def evaluate_system(self, df_full, k_values=[1, 3, 5, 10, 20], sample_size=None):
        """
        Comprehensive evaluation that excludes both:
        1. Resumes with no related jobs
        2. Retrieved jobs with no related resumes
        
        Args:
            df_full: The full dataframe containing resume and job information
            k_values: List of k-values to evaluate for top-k retrieval
            sample_size: Number of resumes to randomly sample for evaluation (None = use all valid resumes)
            
        Returns:
            Dictionary containing evaluation results for each k value
        """
        print("Starting comprehensive evaluation...")
        
        # Find resumes that have at least one related job
        valid_resume_ids = []
        resumes_with_no_related_jobs = 0
        
        unique_resumes = df_full.drop_duplicates(subset=["resume_id"])[["resume_id", "resume_text", "resume_category"]]
        total_resumes = len(unique_resumes)
        
        print("Identifying resumes with related jobs...")
        for resume_id in tqdm(unique_resumes["resume_id"], desc="Filtering resumes"):
            # Get all related jobs for this resume
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
        print(f"Found {len(jobs_with_related_resumes)} jobs with at least one related resume")
        
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
            
            relevant_categories = []
            for _, job_row in related_jobs_df.iterrows():
                if 'resume_category' in job_row and pd.notna(job_row['resume_category']):
                    category = str(job_row['resume_category']).lower().strip()
                    if category and category not in relevant_categories:
                        relevant_categories.append(category)
            
            print(f"\n=== Resume ID: {resume_id} | Resume Category: {resume_category} ===")
            print(f"Number of jobs marked as related in dataset: {len(relevant_job_ids)}")
            print(f"Categories of related jobs: {', '.join(relevant_categories) if relevant_categories else 'None'}")
            
            if relevant_job_ids:
                print("Sample related jobs from dataset:")
                for job_id in relevant_job_ids[:3]:  
                    job_info = related_jobs_df[related_jobs_df["job_id"].astype(str) == job_id]
                    if not job_info.empty:
                        job_row = job_info.iloc[0]
                        job_category = str(job_row.get('resume_category', 'Unknown')).lower().strip()
                        print(f"  Related JobID={job_id}, Title={job_row.get('job_title', 'Unknown')}, Category={job_category}")

            larger_k = 50 
            query_result = self.query_jobs(resume_text, top_k=larger_k)
            all_matches = query_result["matches"]

            filtered_matches = []
            excluded_matches = []
            
            for match in all_matches:
                job_id = str(match["metadata"].get("job_id", match["id"]))
                
                if job_id in jobs_with_related_resumes:
                    filtered_matches.append(match)
                else:
                    excluded_matches.append(match)
                    encountered_unrelated_jobs.add(job_id)
            
            filtered_matches = filtered_matches[:max_k]
            
            print(f"Retrieved matches after filtering: {len(filtered_matches)} (excluded {len(excluded_matches)} jobs with no related resumes)")
            
            for idx, match in enumerate(filtered_matches, 1):
                job_title = match["metadata"].get("job_title", "Unknown")
                job_id = str(match["metadata"].get("job_id", match["id"]))
                job_category = str(match["metadata"].get("resume_category", "Unknown")).lower().strip()
                score = match["score"]
                
                # Determine if this is a correct match (explicit job ID or category matches)
                is_relevant_job = job_id in relevant_job_ids
                is_relevant_category = job_category in relevant_categories and relevant_categories
                is_correct = is_relevant_job or is_relevant_category
                
                match_marker = "*" if is_correct else " "
                
                print(f" {match_marker}Match {idx}: JobID={job_id}, Title={job_title}, Category={job_category}, Score={score:.3f}")
                if is_correct:
                    if is_relevant_job:
                        print(f"    ✓ This job is explicitly marked as relevant in the dataset")
                    if is_relevant_category:
                        print(f"    ✓ Category '{job_category}' matches one of the related job categories: {', '.join(relevant_categories)}")

            # Evaluate for each top-k using the filtered subset
            for k in k_values:
                if k <= len(filtered_matches):
                    top_k_matches = filtered_matches[:k]
                    match_found = False

                    for match in top_k_matches:
                        job_id = str(match["metadata"].get("job_id", match["id"]))
                        job_category = str(match["metadata"].get("resume_category", "Unknown")).lower().strip()

                        # A match is found if either:
                        # 1. The job ID is explicitly marked as related
                        # 2. The job's category matches any of the categories of related jobs
                        if job_id in relevant_job_ids or (job_category in relevant_categories and relevant_categories):
                            match_found = True
                            break

                    results[k]["total_evaluated"] += 1
                    if match_found:
                        results[k]["correct_matches"] += 1

        # Report overall statistics
        print("\n====== Evaluation Statistics ======")
        print(f"Total resumes with at least one related job: {len(valid_resume_ids)}")
        print(f"Total jobs with at least one related resume: {len(jobs_with_related_resumes)}")
        print(f"Total jobs excluded from evaluation (no related resumes): {len(encountered_unrelated_jobs)}")
        
        # Compute success rates
        print("\n====== Evaluation Results ======")
        print("Top-k | Success Rate | Correct Matches | Total Evaluated")
        print("-" * 60)
        for k in k_values:
            correct = results[k]["correct_matches"]
            total = results[k]["total_evaluated"] 
            success_rate = correct / total if total > 0 else 0.0
            results[k]["success_rate"] = success_rate
            print(f"Top-{k:<3} | {success_rate:.4f}       | {correct:<15} | {total}")

        return results

def main():
    print("Starting Job Matching System...")
    system = JobMatchingSystem(CSV_PATH, PERSIST_DIR, COLLECTION_NAME)
    df_full, df_jobs = system.load_and_process_jobs()

    print("\nEvaluating system using 5 random resumes from dataset...")
    evaluation_results = system.evaluate_system(df_full, k_values=[1, 3, 5, 10, 20])

    print("\n====== Evaluation Results ======")
    print("Top-k | Success Rate | Correct Matches | Total Evaluated")
    print("-" * 60)
    for k, metrics in evaluation_results.items():
        print(f"Top-{k:<3} | {metrics['success_rate']:.4f}       | {metrics['correct_matches']}              | {metrics['total_evaluated']}")


if __name__ == "__main__":
    main()
