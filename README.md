# Evaluation Scripts and Resources

## Evaluation Scripts

- `evaluation_bert.py`  
  Evaluates the approach using **BERT embeddings** + **Cosine Similarity**.

- `evaluation_l2.py`  
  Evaluates the approach using **Nomic Embed-Text embeddings** + **L2 (Euclidean) Similarity**.

- `evaluation_main.py`  
  Evaluates the approach using **Nomic Embed-Text embeddings** + **Cosine Similarity** (main approach).

## Dataset

- `resume_job_dataset_updated`  
  The **Resume-Job matching dataset** used in the evaluation phase after cleaning, preprocessing and mergin the initial resumes and jobs datasets included in the presentation and report.

## RAG Implementation

- `app/`  
  Contains the **RAG (Retrieval-Augmented Generation) implementation** using the main approach (`evaluation_main.py`).
  
  Visuals and illustrations of this implementation are included in the project report.
