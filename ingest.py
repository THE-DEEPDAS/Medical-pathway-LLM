import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import time

def ingest_docs():
    try:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(ROOT_DIR, 'Data')
        
        # Initialize embeddings with timeout settings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        # Add configuration for Qdrant
        optimizer_config = {
            "max_optimization_threads": 4
        }
        strict_mode_config = {
            "enabled": False
        }
        
        # Initialize Qdrant client with timeout
        client = QdrantClient(
            url="http://localhost:6333",
            timeout=600.0,  # 10 minutes timeout
            optimizer_config=optimizer_config,
            strict_mode=strict_mode_config
        )
        
        # Create collection with explicit parameters
        collection_name = "medical_db"
        try:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        except Exception as e:
            print(f"Collection creation error: {e}")
            return False

        # Load and process documents in batches
        documents = []
        batch_size = 5  # Reduce batch size
        chunk_size = 300  # Smaller chunks
        
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
        total_docs = len(pdf_files)
        processed = 0
        
        for pdf_file in pdf_files:
            file_path = os.path.join(DATA_DIR, pdf_file)
            pdf = PdfReader(file_path)
            
            for i in range(0, len(pdf.pages), batch_size):
                batch = pdf.pages[i:i + batch_size]
                for page in batch:
                    text = page.extract_text()
                    if text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={"source": file_path, "page": i}
                        )
                        documents.append(doc)
                
                # Process batch
                if documents:
                    try:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,  # Smaller chunks
                            chunk_overlap=50,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        texts = text_splitter.split_documents(documents)
                        
                        # Create vector store in batches
                        for i in range(0, len(texts), 50):  # Process 50 chunks at a time
                            batch_texts = texts[i:i + 50]
                            Qdrant.from_documents(
                                documents=batch_texts,
                                embedding=embeddings,
                                url="http://localhost:6333",
                                prefer_grpc=False,
                                collection_name=collection_name,
                                force_recreate=False  # Don't recreate for each batch
                            )
                            time.sleep(1)  # Small delay between batches
                            
                        documents = []  # Clear processed documents
                        
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        continue
            
            processed += 1
            print(f"Processed {processed}/{total_docs} files")
            time.sleep(2)  # Add delay between files
        
        print("Successfully created Qdrant collection: medical_db")
        return True
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return False

if __name__ == "__main__":
    print("Starting document ingestion...")
    ingest_docs()
