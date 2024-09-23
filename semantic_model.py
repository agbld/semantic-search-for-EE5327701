"""
This file contains the functions to load and infer the distilled, quantized EcomBERT model.
"""

import torch
import time
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from tqdm.autonotebook import trange
from typing import List, Union
def _text_length(text: Union[List[int], List[List[int]]]):
    if isinstance(text, dict):  # {key: value} case
        return len(next(iter(text.values())))
    elif not hasattr(text, "__len__"):  # Object has no len() method
        return 1
    elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
        return len(text)
    else:
        return sum([len(t) for t in text])  # Sum of length of individual strings
    
def inference(tokenizer, model, sentences, batch_size, verbose=False):
    
    length_sorted_idx = np.argsort([-_text_length(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

    embeddings = []
    time_per_batch = []
    with tqdm(total=len(sentences), desc="Batches", disable = not verbose) as pbar:
        for i in trange(0, len(sentences), batch_size, desc="Batches", disable = True):
            start_time = time.time()
            batch = sentences_sorted[i:i+batch_size]
            encoded_inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(torch.device('cpu'))
            with torch.no_grad():
                output = model(**encoded_inputs)['last_hidden_state'].detach()
                batch_prototypes = torch.mean(output, dim=1)
                batch_prototypes = torch.nn.functional.normalize(batch_prototypes, p=2, dim=1).to(torch.device('cpu'))
                embeddings.extend(batch_prototypes)
            time_per_batch.append(time.time() - start_time)
            pbar.update(len(batch))

    embeddings = [embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    embeddings = np.asarray([emb.numpy() for emb in embeddings])

    return embeddings, time_per_batch

def get_semantic_model(model_id: str = './save_model/ABRSS_student_L2_onnx_QINT8'):
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

if __name__ == "__main__":
    """
    Following code will take csv files under ./items and make embeddings for each item and save them as .npy files.
    """

    print("Loading model and tokenizer...")
    model, tokenizer = get_semantic_model()
    print("Model and tokenizer loaded successfully.")

    csv_files = [file for file in os.listdir('./random_samples_1M') if file.endswith('.csv')]
    print(f"Found {len(csv_files)} csv files under ./random_samples_1M")

    with tqdm(total=len(csv_files), desc="Inferencing embeddings") as pbar:
        for csv_file in csv_files:
            items_df = pd.read_csv(f'./random_samples_1M/{csv_file}')
            items_df['product_name'] = items_df['product_name'].astype(str)

            # debug
            items_df = items_df.head(100)

            product_names = items_df['product_name'].values
            
            embeddings, _ = inference(tokenizer, model, product_names, 32, verbose=True)
            embeddings = embeddings.astype(np.float16)  
            
            if not os.path.exists('./embeddings'):
                os.makedirs('./embeddings')
            # if not os.path.exists(f'./embeddings/ecombert'):
            #     os.makedirs(f'./embeddings/ecombert')
            if not os.path.exists(f'./embeddings/semantic_model'):
                os.makedirs(f'./embeddings/semantic_model')
            np.save(f'./embeddings/semantic_model/{csv_file[:-4]}.npy', embeddings)
        
        pbar.update(1)

    print("Embeddings saved successfully.")