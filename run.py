import subprocess
import requests
import time
import os
from ingest import ingest_docs

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)  # Change working directory to script location

def check_qdrant():
    try:
        response = requests.get("http://localhost:6333/collections")
        return response.status_code == 200
    except:
        return False

def check_data_directory():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, 'Data')  # Note the capital 'D' in Data
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}. Please add your medical PDF files.")
        return False
    
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}. Please add medical PDF files.")
        return False
    return True

def main():
    # Step 1: Check Qdrant
    if not check_qdrant():
        print("Error: Qdrant is not running!")
        print("Please start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        return

    # Step 2: Check data directory
    if not check_data_directory():
        return

    # Step 3: Ingest documents
    print("Ingesting documents into Qdrant...")
    if not ingest_docs():
        print("Error during document ingestion!")
        return
    
    print("Document ingestion complete. Starting the application...")
    # Step 4: Start the FastAPI application
    subprocess.run(["uvicorn", "rag:app", "--reload"])

if __name__ == "__main__":
    main()
