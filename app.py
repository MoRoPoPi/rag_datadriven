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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mistralai import Mistral
from pydantic import BaseModel

from query import query_index

PERSIST_DIR = "./chroma_db"  
COLLECTION_NAME = "postings"
EMBED_MODEL_NAME = "nomic-embed-text"
UPLOAD_DIR = "static/uploads"
CSV_PATH = "postings.csv"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "54BCzYTzTa9cwJ8824S8wGsKmpmi9d1c")

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
    
    # Validate page number
    if page > total_pages and total_pages > 0:
        page = 1
    
    # Paginate the results
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_jobs)
    
    jobs_page = df.iloc[start_idx:end_idx].copy()
    
    # Add row index as row_id for each job
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
    
    # Check if the row_id is valid
    if row_id < 0 or row_id >= len(df):
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get the job by its row index
    job = df.iloc[row_id].to_dict()
    
    # Add the row_id to the job data
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
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, resume.filename)
        
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)
        
        resume_text = ""
        # Try OCR with Mistral if API key is available
        if MISTRAL_API_KEY:
            try:
                # Initialize Mistral client
                client = Mistral(api_key=MISTRAL_API_KEY)
                
                # Encode file to base64 for Mistral OCR
                with open(temp_file_path, "rb") as file:
                    base64_file = base64.b64encode(file.read()).decode('utf-8')
                
                # Determine document type based on file extension
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
        
        # Simple query to the index with the resume text
        results = query_index(resume_text, return_nodes=True, similarity_top_k=similarity_top_k)
        
        # Convert to presentable format
        matched_jobs = []
        if results:
            df = load_job_postings()
            for node in results:
                # Extract job_id from metadata if available
                job_id = node.node.metadata.get('csv_row_index')
                if job_id is not None:
                    job_data = df.iloc[job_id].to_dict() if job_id < len(df) else None
                    if job_data:
                        # Add row_id to the job data
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
                "resume_text": resume_text,
                "matched_jobs": matched_jobs
            }
        )

# Run app with uvicorn
if __name__ == "__main__":
    # Ensure the upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Run the FastAPI app
    uvicorn.run("app:app", reload=True)