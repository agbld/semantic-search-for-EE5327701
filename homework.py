#%%
import pandas as pd
import os

from get_top_100_queries import find_top_k_queries
import glob

import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--student_id", type=str, default='M11207314', help='Student ID for homework mode.')
parser.add_argument('--assigned_queries', type=str, default='./Mxxxxxxxx_xxx_assigned_queries.csv', help='Path to the assigned queries CSV file.')

args = parser.parse_args()

# Find top 100 queries
print('Loading assigned queries...')
queries_df = pd.read_csv(args.assigned_queries, usecols=['key_word'])
queries_df.columns = ['name']

#%%
result_dfs = []
for i in range(len(queries_df)):
    query = queries_df['name'][i]
    results = {
        '搜尋詞': query,
        'Rank': list(range(1, 11)),
    }
    result_df = pd.DataFrame.from_dict(results).reset_index()
    result_dfs.append(result_df)
    
# Save the results to a CSV file
results_folder = './tmp'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
result_dfs = pd.concat(result_dfs)
result_dfs.to_csv(f'{results_folder}/base.csv', index=False)

#%%
# Load the CSV files
base = pd.read_csv('./tmp/base.csv')
tfidf_df = pd.read_csv('./tmp/tf-idf.csv')
semantic_df = pd.read_csv('./tmp/semantic_model.csv')

final_df = {
    '搜尋詞': base['搜尋詞'].tolist(),
    'Rank': base['Rank'].tolist(),
    'tf-idf': tfidf_df['tf-idf'].tolist(),
    'ner_relevancy_0': '',
    'semantic_model': semantic_df['semantic_model'].tolist(),
    'ner_relevancy_1': '',
}

final_df = pd.DataFrame.from_dict(final_df)

final_df.to_excel(f'{args.student_id}_搜尋比較.xlsx', index=False)

#%%