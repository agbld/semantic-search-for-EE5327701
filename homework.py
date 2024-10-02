#%%
import pandas as pd
import os

from get_top_100_queries import find_top_k_queries
import glob

# Find top 100 queries
print('Finding top 100 queries...')
queries_df = find_top_k_queries('./items', 100)

#%%
result_dfs = []
for i in range(len(queries_df)):
    query = queries_df['name'][i]
    print(f'Searching for query: {query} ({i+1}/{len(queries_df)})')
    csv_files = glob.glob(f'./items/*_{query}.csv')
    if csv_files:
        csv_file = csv_files[0]
    else:
        print(f'No CSV file found for query: {query}')
        continue
    df = pd.read_csv(csv_file, usecols=['product_name'], nrows=10)
    results = {
        '搜尋詞': query,
        'Rank': list(range(1, 11)),
        'Coupang': df['product_name'].tolist(),
    }
    result_df = pd.DataFrame.from_dict(results).reset_index()
    result_dfs.append(result_df)
    
# Save the results to a CSV file
results_folder = './tmp'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
result_dfs = pd.concat(result_dfs)
result_dfs.to_csv(f'{results_folder}/coupang.csv', index=False)

#%%
# Load the CSV files
coupang_df = pd.read_csv('./tmp/coupang.csv')
tfidf_df = pd.read_csv('./tmp/tf-idf.csv')
semantic_df = pd.read_csv('./tmp/semantic_model.csv')

final_df = {
    '搜尋詞': coupang_df['搜尋詞'].tolist(),
    'Rank': coupang_df['Rank'].tolist(),
    'Coupang': coupang_df['Coupang'].tolist(),
    'Coupang_relevancy': '',
    'tf-idf': tfidf_df['tf-idf'].tolist(),
    'tf-idf_relevancy': '',
    'semantic_model': semantic_df['semantic_model'].tolist(),
    'semantic_model_relevancy': '',
}

final_df = pd.DataFrame.from_dict(final_df)

final_df.to_excel(f'M11207314_搜尋比較.xlsx', index=False)

#%%