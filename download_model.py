import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    path = os.path.join('models', filename)
    
    print(f"Downloading {filename}...")
    print(f"File size: {total_size / (1024*1024*1024):.2f} GB")
    
    block_size = 1024*1024  # 1MB chunks
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    
    progress_bar.close()
    print(f"Download completed! File saved to: {path}")

if __name__ == "__main__":
    model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    filename = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    
    download_file(model_url, filename)
