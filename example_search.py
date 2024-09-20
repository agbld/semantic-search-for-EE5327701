import time
import requests
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="ecombert", choices=["ecombert", "ckipbert"],
                    help="Type of model to use: 'ecombert' or 'ckipbert'")
args = parser.parse_args()

# Set the embeddings directory based on model type
embeddings_dir = f'./embeddings/{args.model_type}/'

# Load pre-computed product names and embeddings
product_names = []
product_embeddings = []

# Ensure the embeddings directory exists
if not os.path.exists(embeddings_dir):
    raise FileNotFoundError(f"Embeddings directory '{embeddings_dir}' not found.")

# Loop through all .npy files in the embeddings directory
for file in os.listdir(embeddings_dir):
    if file.endswith('.npy'):
        embedding_file = os.path.join(embeddings_dir, file)
        csv_file = os.path.join('./items', file.replace('.npy', '.csv'))

        # Check if the corresponding CSV file exists
        if not os.path.exists(csv_file):
            continue

        # Load product names from the CSV file
        items_df = pd.read_csv(csv_file)
        product_names.extend(items_df['product_name'].values)

        # Load product embeddings from the .npy file
        embeddings = np.load(embedding_file)
        product_embeddings.append(embeddings)

# Concatenate all embeddings into a single numpy array
product_embeddings = np.concatenate(product_embeddings, axis=0)

print(f'Number of products: {len(product_names)}')
print(f'Number of pre-computed embeddings: {product_embeddings.shape[0]}')

# Function to get embeddings from the server
def get_embeddings(text: list, url: str = 'http://localhost:5000/api/embed') -> list:
    headers = {'Content-Type': 'application/json'}
    data = {'text': text}

    response = requests.post(url, json=data, headers=headers)
    return response.json()

# Function to search for the top k items
def search(query, product_names, product_embeddings, top_k=5):
    product_names_series = pd.Series(product_names)

    # Get the embedding for the query via API call
    query_embeddings = get_embeddings([query])

    # Convert query_embeddings to numpy array
    query_embeddings = np.array(query_embeddings)

    # Calculate cosine similarity between query and product embeddings
    scores = cosine_similarity(query_embeddings, product_embeddings)

    # Get indices of the top k most similar products
    top_k_indices = np.argsort(-scores[0])[:top_k]
    top_k_names = product_names_series.iloc[top_k_indices]
    top_k_scores = scores[0][top_k_indices]

    return top_k_names, top_k_scores

# Run in interactive mode
while True:
    query = input('Enter query (type "exit" to quit): ')
    if query.lower() == 'exit':
        break

    start_time = time.time()
    top_k_names, scores = search(query, product_names, product_embeddings)
    elapsed_time = time.time() - start_time
    print(f'Took {elapsed_time:.4f} seconds to search')

    for i, (name, score) in enumerate(zip(top_k_names, scores)):
        print(f'[Rank {i+1} | Score: {score:.4f}] {name}')
