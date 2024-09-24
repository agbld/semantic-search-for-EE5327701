from huggingface_hub import hf_hub_download
import os
import shutil
import zipfile

# Function to unzip the file and delete the zip file
def unzip_and_remove(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())
        print(f"Unzipped {zip_file_path} to current directory.")
    os.remove(zip_file_path)
    print(f"Deleted zip file: {zip_file_path}")

# Function to move specific files to ./embeddings folder
def move_to_embeddings_folder(file_path):
    embeddings_folder = os.path.join(os.getcwd(), 'embeddings')
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
        print(f"Created embeddings folder at {embeddings_folder}")
    destination_path = os.path.join(embeddings_folder, os.path.basename(file_path))
    shutil.move(file_path, destination_path)
    print(f"Moved {file_path} to {destination_path}")

# Check if the unzipped folder exists before downloading
def download_if_not_exists(repo_id, filename, folder_name):
    if os.path.exists(folder_name):
        print(f"Folder {folder_name} already exists. Skipping download.")
        return None
    else:
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

# Define the embeddings folder paths
semantic_model_folder = os.path.join(os.getcwd(), 'embeddings', 'semantic_model')
ckipbert_folder = os.path.join(os.getcwd(), 'embeddings', 'ckipbert')

# Download semantic_model.zip if it doesn't exist in embeddings folder
semantic_model_path = download_if_not_exists(
    repo_id="clw8998/Semantic-Search-dataset-for-EE5327701", 
    filename="semantic_model.zip", 
    folder_name=semantic_model_folder
)

# Download ckipbert.zip if it doesn't exist in embeddings folder
ckipbert_path = download_if_not_exists(
    repo_id="clw8998/Semantic-Search-dataset-for-EE5327701", 
    filename="ckipbert.zip", 
    folder_name=ckipbert_folder
)

# Download random_samples_1M.zip if it doesn't exist in the current directory
random_samples_1M_path = download_if_not_exists(
    repo_id="clw8998/Semantic-Search-dataset-for-EE5327701", 
    filename="random_samples_1M.zip", 
    folder_name='./random_samples_1M'
)

# Move and unzip semantic_model.zip if it was downloaded
if semantic_model_path:
    unzip_and_remove(semantic_model_path)
    move_to_embeddings_folder('./semantic_model')

# Move and unzip ckipbert.zip if it was downloaded
if ckipbert_path:
    unzip_and_remove(ckipbert_path)
    move_to_embeddings_folder('./ckipbert')

# Unzip random_samples_1M.zip if it was downloaded
if random_samples_1M_path:
    unzip_and_remove(random_samples_1M_path)
