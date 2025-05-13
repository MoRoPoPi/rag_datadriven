import base64
import os
import shutil
import tempfile
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mistralai import Mistral
import ollama

from query import query_index

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "postings"
EMBED_MODEL_NAME = "nomic-embed-text"
UPLOAD_DIR = "static/uploads"
CSV_PATH = "postings.csv"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "54BCzYTzTa9cwJ8824S8wGsKmpmi9d1c")
OLLAMA_MODEL = "llama3.2" 

app = FastAPI(title="Job Postings RAG System")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class JobPosting(BaseModel):
    job_id: str
    company_name: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    formatted_work_type: Optional[str] = None
    formatted_experience_level: Optional[str] = None
    skills_desc: Optional[str] = None
    remote_allowed: Optional[bool] = None
    job_posting_url: Optional[str] = None

class SearchResult(BaseModel):
    job: JobPosting
    score: float

# Summarize using Ollama
def summarize_with_ollama(text: str, model: str = OLLAMA_MODEL) -> str:
    prompt = f"Summarize the following resume text concisely within 2048 words directly, without adding any introductory phrases like 'Here is a summary...', and provide the summary as paragraphs, not bullet points:\n\n{text}" 
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        # print(f"Response from Ollama:")  
        # print(f"Full response from Ollama: {response}")  
        return response['message']['content']
    except Exception as e:
        print(f"Ollama summarization error: {e}")
        return text  

def summarize_resume(resume_text: str) -> str:
    """
    Summarize the resume text using Ollama LLM
    """
    try:
        return summarize_with_ollama(resume_text, model=OLLAMA_MODEL)
    except Exception as e:
        print(f"Error during summarization: {e}")
        return resume_text  

def load_job_postings():
    try:
        df = pd.read_csv(CSV_PATH)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(
    request: Request, 
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50)
):
    df = load_job_postings()
    total_jobs = len(df)
    total_pages = (total_jobs + page_size - 1) // page_size
    
    if page > total_pages and total_pages > 0:
        page = 1
    
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_jobs)
    
    jobs_page = df.iloc[start_idx:end_idx].copy()
    
    jobs_records = []
    for i, job in enumerate(jobs_page.to_dict('records')):
        job_with_index = job.copy()
        job_with_index['row_id'] = start_idx + i
        jobs_records.append(job_with_index)
    
    return templates.TemplateResponse(
        "jobs.html", 
        {
            "request": request,
            "jobs": jobs_records,
            "total_jobs": total_jobs,
            "current_page": page,
            "total_pages": total_pages,
            "page_size": page_size
        }
    )

@app.get("/job/{row_id}", response_class=HTMLResponse)
async def job_detail(request: Request, row_id: int):
    df = load_job_postings()
    
    if row_id < 0 or row_id >= len(df):
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = df.iloc[row_id].to_dict()
    
    job['row_id'] = row_id
    
    return templates.TemplateResponse(
        "job_detail.html", 
        {
            "request": request,
            "job": job
        }
    )

@app.get("/resume", response_class=HTMLResponse)
async def resume_upload_page(request: Request):
    return templates.TemplateResponse("resume_upload.html", {"request": request})

@app.post("/resume/match", response_class=HTMLResponse)
async def match_resume(
    request: Request,
    resume: UploadFile = File(...),
    similarity_top_k: int = Query(20, ge=1, le=20, description="Number of job matches to return")
):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, resume.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)
        
        resume_text = ""
        if MISTRAL_API_KEY:
            try:
                client = Mistral(api_key=MISTRAL_API_KEY)
                
                with open(temp_file_path, "rb") as file:
                    base64_file = base64.b64encode(file.read()).decode('utf-8')
                
                file_extension = resume.filename.lower().split('.')[-1]
                document_type = "document_url" if file_extension == "pdf" else "image_url"
                content_type = "application/pdf" if file_extension == "pdf" else f"image/{file_extension}"
                
                # Process document with Mistral OCR
                ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": document_type,
                        "document_url": f"data:{content_type};base64,{base64_file}"
                    }
                )
                
                # Extract text from OCR response
                resume_text = "\n".join([page.markdown for page in ocr_response.pages])
                
            except Exception as e:
                print(f"OCR processing error: {e}")
                raise HTTPException(status_code=500, detail="Error processing resume with OCR")
        
        # Summarize the resume before embedding
        original_resume_text = resume_text
        summarized_resume = summarize_resume(resume_text)
        # print(f"Summarized resume text: {summarized_resume}...")  
        results = query_index(summarized_resume, return_nodes=True, similarity_top_k=similarity_top_k)
        
        # Convert to presentable format
        matched_jobs = []
        if results:
            df = load_job_postings()
            for node in results:
                job_id = node.node.metadata.get('csv_row_index')
                if job_id is not None:
                    job_data = df.iloc[job_id].to_dict() if job_id < len(df) else None
                    if job_data:
                        job_data['row_id'] = job_id
                        matched_jobs.append({
                            "job": job_data,
                            "score": node.get_score(),
                            "relevance": f"{node.score:.2f}"
                        })
        
        return templates.TemplateResponse(
            "resume_results.html", 
            {
                "request": request,
                "resume_text": original_resume_text,
                "summarized_resume": summarized_resume,
                "matched_jobs": matched_jobs,
                "summarization_method": "Ollama (" + OLLAMA_MODEL + ")"
            }
        )

if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    uvicorn.run("app:app", reload=True)
