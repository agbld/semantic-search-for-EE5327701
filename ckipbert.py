from transformers import BertTokenizerFast, AutoModel
import torch
import time
import numpy as np
from tqdm import tqdm
from tqdm.autonotebook import trange
import pandas as pd
import os

def get_ckipbert(model_id: str = './save_model/ckiplab-bert-base-chinese'):
    tokenizer = BertTokenizerFast.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    return model, tokenizer

def inference(tokenizer, model, sentences: list, batch_size=16, verbose=False):
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
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

if __name__ == "__main__":
    """
    Following code will take csv files under ./items and make embeddings for each item and save them as .npy files.
    """

    print("Loading model and tokenizer...")
    model, tokenizer = get_ckipbert()
    print("Model and tokenizer loaded successfully.")

    csv_files = [file for file in os.listdir('./items') if file.endswith('.csv')]
    print(f"Found {len(csv_files)} csv files under ./items")

    with tqdm(total=len(csv_files), desc="Inferencing embeddings") as pbar:
        for csv_file in csv_files:
            items_df = pd.read_csv(f'./items/{csv_file}')

            # debug
            items_df = items_df.head(100)

            product_names = items_df['product_name'].values
            
            embeddings, _ = inference(tokenizer, model, product_names, 32, verbose=True)
            
            if not os.path.exists('./embeddings'):
                os.makedirs('./embeddings')
            if not os.path.exists(f'./embeddings/ckipbert'):
                os.makedirs(f'./embeddings/ckipbert')
            np.save(f'./embeddings/ckipbert/{csv_file[:-4]}.npy', embeddings)
            
            pbar.update(1)

    print("Embeddings saved successfully.")