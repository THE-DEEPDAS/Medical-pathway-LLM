import subprocess
import sys
import os
import requests
import time

# Add at the start of the file, before other code
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)  # Change working directory to script location

def check_requirements():
    """Check if all required components are available"""
    # Use absolute path for Data directory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, 'Data')  # Note the capital 'D' in Data
    
    # Check if data directory has PDFs
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}. Please add some medical PDFs.")
        return False

    # Check if models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory.")

    # Check if Qdrant is running
    try:
        response = requests.get("http://localhost:6333/collections")
        if response.status_code != 200:
            raise Exception("Qdrant not responding properly")
    except Exception as e:
        print("Error: Qdrant is not running!")
        print("Please start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        return False

    return True

def main():
    try:
        # Check all requirements
        if not check_requirements():
            return

        # Import ingest only after requirements are checked
        from ingest import ingest_docs
        
        # Ingest documents
        print("Ingesting documents...")
        if not ingest_docs():
            print("Failed to ingest documents. Please check the data directory.")
            return

        # Start the FastAPI server
        print("Starting the application...")
        subprocess.run(["uvicorn", "rag:app", "--reload"], check=True)
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except subprocess.CalledProcessError:
        print("Error starting the server. Make sure all requirements are installed.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
