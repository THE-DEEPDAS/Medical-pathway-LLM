import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader
from langchain.docstore.document import Document  # Add this import

def ingest_docs():
    try:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(ROOT_DIR, 'Data')
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Check for PDFs using absolute path
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            raise FileNotFoundError(f"Data directory created at {DATA_DIR}. Please add PDF files.")
            
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {DATA_DIR}!")
        
        # Modified PDF loading section
        documents = []
        for pdf_file in pdf_files:
            file_path = os.path.join(DATA_DIR, pdf_file)
            pdf = PdfReader(file_path)
            for page in pdf.pages:
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    # Create proper Document objects
                    doc = Document(
                        page_content=text,
                        metadata={"source": file_path}
                    )
                    documents.append(doc)
            
        print(f"Loaded {len(documents)} pages from PDFs")
        
        if not documents:
            raise ValueError("No documents were loaded!")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        print(f"Split into {len(texts)} chunks")
        
        if not texts:
            raise ValueError("No text chunks were created!")
        
        # Create Qdrant vector store
        url = "http://localhost:6333"
        qdrant = Qdrant.from_documents(
            documents=texts,
            embedding=embeddings,
            url=url,
            prefer_grpc=False,
            collection_name="medical_db",
            force_recreate=True  # This will recreate the collection if it exists
        )
        
        print("Successfully created Qdrant collection: medical_db")
        return True
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return False

if __name__ == "__main__":
    print("Starting document ingestion...")
    ingest_docs()
