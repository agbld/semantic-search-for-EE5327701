import time
import requests
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-computed product names and embeddings from ./items folder
product_names = []
product_embeddings = []
for file in os.listdir('./items'):
    if file.endswith('.npy'):
        # Check if the embedding matrix has a .csv counterpart, if not, skip
        csv_file = file.replace('.npy', '')
        if not os.path.exists(f'./items/{csv_file}.csv'):
            continue

        # Load product names
        items_df = pd.read_csv(f'./items/{csv_file}.csv')
        product_names.extend(items_df['product_name'].values)

        # Load product embeddings (2D numpy array)
        embeddings = np.load(f'./items/{file}')
        product_embeddings.append(embeddings)
product_embeddings = np.concatenate(product_embeddings, axis=0)

print(f'Number of products: {len(product_names)}')
print(f'Number of pre-computed embeddings: {product_embeddings.shape[0]}')

# Function to get embeddings from the server
def get_embeddings(text: list, url: str = 'http://localhost:5000/api/embed') -> list:
    url = 'http://localhost:5000/api/embed'
    headers = {'Content-Type': 'application/json'}
    data = {'text': text}

    response = requests.post(url, json=data, headers=headers)
    return response.json()

# Function to search for the top k items
def search(query, product_names, product_embeddings, top_k=5):
    product_names = pd.Series(product_names)

    # Get the embedding for the query with API call
    query_embeddings = get_embeddings([query])

    # Calculate cosine similarity between query and product embeddings and get the top k items
    scores = cosine_similarity(query_embeddings, product_embeddings)
    top_k_indices = np.argsort(-scores[0])[:top_k]
    top_k_names = product_names[top_k_indices]
    top_k_scores = scores[0][top_k_indices]

    return top_k_names, top_k_scores

# Run in interactive mode
while True:
    query = input('Enter query: ')
    if query == 'exit':
        break

    start = time.time()
    top_k_names, scores = search(query, product_names, product_embeddings)
    print(f'Takes {time.time() - start:.4f} seconds to search')

    for i, name in enumerate(top_k_names):
        print(f'[Rank {i+1} ({round(scores[i], 4)})] {name}')