
import os
import requests
import tarfile
import zipfile
import subprocess
import sys

DATA_DIR = "c:\\Users\\sahin\\.gemini\\antigravity\\projects\\data"

DATASETS = {
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "fruits360": "moltean/fruits",
    "animals10": "alessiocorrado99/animals10"
}

def create_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
    else:
        print(f"Directory exists: {DATA_DIR}")

# def download_cifar():
#     url = DATASETS["cifar10"]
#     filename = url.split("/")[-1]
#     filepath = os.path.join(DATA_DIR, filename)
    
#     if os.path.exists(filepath):
#         print(f"{filename} already exists. Skipping.")
#         return

#     print(f"Downloading {filename}...")
#     try:
#         response = requests.get(url, stream=True)
#         with open(filepath, 'wb') as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print("Download complete.")
        
#         # Extract
#         print("Extracting...")
#         with tarfile.open(filepath, "r:gz") as tar:
#             tar.extractall(path=DATA_DIR)
#         print("Extraction complete.")
        
#     except Exception as e:
#         print(f"Error downloading CIFAR-10: {e}")

def check_kaggle_auth():
    # Check if kaggle.json exists in standard location or env vars are set
    kaggle_config_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json = os.path.join(kaggle_config_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        return True
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        return True
    return False

def download_kaggle_dataset(dataset_name):
    print(f"Attempting to download Kaggle dataset: {dataset_name}")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", DATA_DIR, "--unzip"], check=True)
        print(f"Successfully downloaded {dataset_name}")
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it (pip install kaggle) and ensure it's in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {dataset_name}: {e}")
        print("Make sure you have set up your Kaggle API key (~/.kaggle/kaggle.json).")

def main():
    create_dir()
    download_cifar()
    
    if check_kaggle_auth():
        download_kaggle_dataset(DATASETS["fruits360"])
        download_kaggle_dataset(DATASETS["animals10"])
    else:
        print("\nSkipping Kaggle datasets (auth not found).")
        print("To download Fruits-360 and Animals-10, please set up Kaggle API keys.")

if __name__ == "__main__":
    main()
