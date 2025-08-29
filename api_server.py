from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
from original import summarize_document_with_kmeans_clustering, embeddings, llm

app = FastAPI()

@app.post("/summarize")
async def summarize(pdf_path: str = Form(None), chunk_size: int = Form(2000), num_clusters: int = Form(20), file: UploadFile = File(None)):
    # Handle uploaded file
    if file:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        pdf_path = temp_path

    summary = summarize_document_with_kmeans_clustering(pdf_path, llm, embeddings)
    return JSONResponse({"summary": summary})