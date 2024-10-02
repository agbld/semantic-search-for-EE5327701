import os
os.environ ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import get_dataset

import time
import numpy as np
import pandas as pd
import argparse
import faiss

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="semantic_model", choices=["semantic_model", "ckipbert"],
                    help="Type of model to use: 'semantic_model' or 'ckipbert'")
parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
parser.add_argument('--homework', action='store_true', help='Run in homework mode.')
args = parser.parse_args()

# Prepare the model and inference function based on the model type
if args.model_type == "semantic_model":
    from semantic_model import get_semantic_model, inference
    model, tokenizer = get_semantic_model()
elif args.model_type == "ckipbert":
    from ckipbert import get_ckipbert, inference
    model, tokenizer = get_ckipbert()

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
        csv_file = os.path.join('./random_samples_1M', file.replace('.npy', '.csv'))

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

# Convert embeddings to float32
product_embeddings = product_embeddings.astype('float32')

# Normalize embeddings for cosine similarity
faiss.normalize_L2(product_embeddings)

# Build FAISS index
embedding_dim = product_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)  # Using Inner Product as similarity measure
index.add(product_embeddings)

print(f'FAISS index built with {index.ntotal} vectors.')

# Convert product names to pandas Series for easy indexing
product_names_series = pd.Series(product_names)

# Function to search for the top k items
def search(query, product_names_series, index, top_k=args.top_k):
    # Get the embedding for the query
    query_embedding, _ = inference(tokenizer, model, [query] , 16)
    query_embedding = np.array([query_embedding]).astype('float32')[0]

    # Normalize query embedding
    faiss.normalize_L2(query_embedding)

    # Search using the index
    scores, indices = index.search(query_embedding, top_k)

    # Retrieve search results
    top_k_names = product_names_series.iloc[indices[0]].values
    top_k_scores = scores[0]

    return top_k_names, top_k_scores

# Run in interactive mode
while True and not args.homework:
    query = input('Enter query (type "exit" to quit): ')
    if query.lower() == 'exit':
        break

    start_time = time.time()
    top_k_names, scores = search(query, product_names_series, index)
    elapsed_time = time.time() - start_time
    print(f'Took {elapsed_time:.4f} seconds to search')

    for i, (name, score) in enumerate(zip(top_k_names, scores)):
        print(f'[Rank {i+1} | Score: {score:.4f}] {name}')

# Run in homework mode
if args.homework:
    """
    See README.md for the homework instructions.
    """

    # Find top 100 queries
    from get_top_100_queries import find_top_k_queries
    print('Finding top 100 queries...')
    queries_df = find_top_k_queries('./items', 100)
    
    # Search for the top k items for each query
    result_dfs = []
    results_folder = './results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    for i in range(len(queries_df)):
        query = queries_df['name'][i]
        print(f'Searching for query: {query} ({i+1}/{len(queries_df)})')

        # For first requirement
        top_k_names, scores = search(query, product_names_series, index, top_k=int(queries_df['count'][i]))
        results = {
            'top_k_names': top_k_names.tolist(),
            'scores': scores.tolist()
        }
        results_df = pd.DataFrame.from_dict(results).reset_index()
        results_df.columns = ['query', 'top_k_names', 'scores']
        results_df.to_csv(os.path.join(results_folder, f'M11207314_semantic_{query}.csv'), index=False, encoding='utf-8')

        # For second requirement
        top_k_names, _ = search(query, product_names_series, index)
        results = {
            '搜尋詞': query,
            'Rank': list(range(1, 11)),
            'semantic_model': top_k_names.tolist(),
        }

        # Convert results to DataFrame
        result_df = pd.DataFrame.from_dict(results).reset_index()
        result_dfs.append(result_df)

    # Compress the results folder to a zip file
    print('Compressing the results folder...')
    import shutil
    shutil.make_archive('./M11207314_results', 'zip', './results')

    # Save the results to a CSV file
    result_dfs = pd.concat(result_dfs)
    if not os.path.exists(f'./tmp/'):
        os.makedirs(f'./tmp/')
    result_dfs.to_csv(f'./tmp/semantic_model.csv', index=False)