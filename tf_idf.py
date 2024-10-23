import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import jieba
import os
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from argparse import ArgumentParser
import pickle

# Command line arguments
try:
    argparser = ArgumentParser()
    argparser.add_argument('items_folder', type=str, help='Folder containing the items (csv files) to search.')
    argparser.add_argument('-k', '--top_k', type=int, default=5, help='Number of top k items to return.')
    argparser.add_argument('-f', '--file_idx', type=int, default=-1, help='File index of activate_item folder. Use -1 to load all files at once.')
    argparser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive mode.')
    argparser.add_argument('-s', '--sample_size', type=int, default=100000, help='Number of items to sample from the dataset for TF-IDF model creation. Use -1 to load all items.')
    argparser.add_argument('-a', '--all', action='store_true', help='Load all items without dropping duplicates.')
    argparser.add_argument('-c', '--create', action='store_true', help='Create the TF-IDF models without using the saved models.')
    argparser.add_argument('--api_server', action='store_true', help='Run in API server mode.')
    argparser.add_argument('--homework', action='store_true', help='Run in homework mode.')
    argparser.add_argument('--student_id', type=str, default='M11207314', help='Student ID for homework mode.')
    argparser.add_argument('--assigned_queries', type=str, default='./Mxxxxxxxx_xxx_assigned_queries.csv', help='Path to the assigned queries CSV file.')

    args = argparser.parse_args()
    items_folder = args.items_folder
    top_k = args.top_k
    file_idx = args.file_idx
    interactive = args.interactive
    sample_size = args.sample_size
    drop_duplicates = not args.all
    create = args.create
except:
    items_folder = './items'
    top_k = 5
    file_idx = -1
    interactive = True
    sample_size = 100000
    drop_duplicates = True
    create = True

# Check if the items folder exists
if not os.path.exists(items_folder):
    print(f'Error: Folder "{items_folder}" not found.')
    response = input(f'Do you want to create the folder "{items_folder}"? (yes/no): ').strip().lower()
    if response == 'yes':
        os.makedirs(items_folder)
        print(f'Folder "{items_folder}" created. Please add the items (csv files) to this folder and rerun the script.')
    else:
        print('Please provide the correct path to the items folder and rerun the script.')
    exit()

if file_idx == -1:
    print(f'Loading all files from: "{items_folder}"')
else:
    print(f'Loading {file_idx}th file from: "{items_folder}"')

# load item file from activate_item folder by file_idx
timer_start = time.time()
if file_idx >= 0:
    path_to_item_file = [file for file in os.listdir(items_folder) if file.endswith('.csv')][file_idx]
    items_df = pd.read_csv(os.path.join(items_folder, path_to_item_file), usecols=['product_name'])
else:
    path_to_item_files = [file for file in os.listdir(items_folder) if file.endswith('.csv')]
    items_df = []
    for file in path_to_item_files:
        try:
            items_df.append(pd.read_csv(os.path.join(items_folder, file), usecols=['product_name']))
        except:
            print(f'Error loading file: {file}')
    print(f'Loaded {len(items_df)} files.')
    items_df = pd.concat(items_df, ignore_index=True)
    path_to_item_file = 'all'

# Sample items from the dataset
if sample_size != -1:
    items_df = items_df.sample(n=sample_size)

# Ensure all product_name entries are strings
items_df['product_name'] = items_df['product_name'].astype(str)

# preprocess item_df
items_df['product_name'] = items_df['product_name'].map(html.unescape)
items_df['product_name'] = items_df['product_name'].fillna('')

if drop_duplicates:
    items_df = items_df.drop_duplicates(subset='product_name')
print(f'Processed {len(items_df)} items in {time.time() - timer_start:.2f} seconds.')

timer_start = time.time()

# Disable jieba cache logging
jieba.setLogLevel(jieba.logging.WARN)
class JiebaTokenizer(object):
    def __init__(self, class_name):
        self.class_name = class_name
        for each_name in self.class_name:
            userdict_path = './Lexicon_merge/{}.txt'.format(each_name)
            if os.path.exists(userdict_path):
                jieba.load_userdict(userdict_path)
            else:
                print(f"User dictionary {userdict_path} not found, skipping.")
    
    def __call__(self, x):
        tokens = jieba.lcut(x, cut_all=False)
        stop_words = ['【','】','/','~','＊','、','（','）','+','‧',' ','']
        tokens = [t for t in tokens if t not in stop_words]
        return tokens

tokenizer = JiebaTokenizer(class_name=['type','brand','p-other'])

# Path to save/load the models
model_path = 'tf_idf_checkpoint.pkl'

# Function to save the models
def save_models_and_matrices(tfidf, items_tfidf_matrix, path):
    with open(path, 'wb') as file:
        pickle.dump({
            'tfidf': tfidf,
            'items_tfidf_matrix': items_tfidf_matrix,
        }, file)

# Function to load the models
def load_models_and_matrices(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data['tfidf'], data['items_tfidf_matrix']

# Check if the models are already saved
if os.path.exists(model_path) and not create:
    # If saved, load the models
    tfidf, items_tfidf_matrix = load_models_and_matrices(model_path)
else:
    # If not saved, create the models
    print('TF-IDF models not found. Creating them...')

    tfidf = TfidfVectorizer(token_pattern=None, tokenizer=tokenizer, ngram_range=(1,2))
    items_tfidf_matrix = tfidf.fit_transform(tqdm(items_df['product_name']))
    
    # tfidf_char = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", analyzer='char')
    # items_tfidf_matrix_char = tfidf_char.fit_transform(items_df['product_name'])

    save_models_and_matrices(tfidf, items_tfidf_matrix, model_path)

print(f'TF-IDF models loaded in {time.time() - timer_start:.2f} seconds.')

# Function to search for the top k items
def search(query, top_k=top_k):
    query_tfidf = tfidf.transform([query]) # sparse array
    scores = cosine_similarity(query_tfidf, items_tfidf_matrix)
    top_k_indices = np.argsort(-scores[0])[:top_k]
    
    # sum_of_score = sum(scores[0])
    # if sum_of_score < 10 : 
    #     query_tfidf = tfidf_char.transform([query]) # sparse array
    #     scores = cosine_similarity(query_tfidf, items_tfidf_matrix_char)
    #     top_k_indices = np.argsort(-scores[0])[:top_k]
    #     sum_of_score = sum(scores[0])
        
    top_k_names = items_df['product_name'].values[top_k_indices]
    top_k_scores = scores[0][top_k_indices]

    return top_k_names, top_k_scores

# Run in interactive mode
if interactive:

    while True:
        query = input('Enter query: ')
        if query == 'exit':
            break
        top_k_names, scores = search(query)

        for i, name in enumerate(top_k_names):
            print(f'[Rank {i+1} ({round(scores[i], 4)})] {name}')

# Run in API server mode. 
# Note: This part is not necessary to run if you are student. It is not required in the assignment.
elif args.api_server:
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route('/search', methods=['GET'])
    def search_api():
        query = request.args.get('query')
        top_k_names, scores = search(query)
        return jsonify({'top_k_names': top_k_names.tolist(), 'scores': scores.tolist()})
    
    app.run(host='0.0.0.0', port=5000)

    # Example usage: http://localhost:5000/search?query=iphone

# Run in homework mode
elif args.homework:
    """
    See README.md for the homework instructions
    """

    # Find top 100 queries
    print('Loading assigned queries...')
    queries_df = pd.read_csv(args.assigned_queries, usecols=['key_word'])
    queries_df.columns = ['name']
    
    # Search for the top k items for each query
    result_dfs = []
    for i in range(len(queries_df)):
        query = queries_df['name'][i]
        query = query.replace('/', ' ')
        print(f'Searching for query: {query} ({i+1}/{len(queries_df)})')
        top_k_names, scores = search(query, top_k=10)
        results = {
            '搜尋詞': query,
            'Rank': list(range(1, 11)),
            'tf-idf': top_k_names.tolist(),
        }

        # Convert results to DataFrame
        result_df = pd.DataFrame.from_dict(results).reset_index()
        result_dfs.append(result_df)

    # Save the results to a CSV file
    results_folder = './tmp'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    result_dfs = pd.concat(result_dfs)
    result_dfs.to_csv(f'{results_folder}/tf-idf.csv', index=False)