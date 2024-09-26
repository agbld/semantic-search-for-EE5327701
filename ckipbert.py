from transformers import BertTokenizerFast, AutoModel
import torch
import time
import numpy as np
from tqdm import tqdm
from tqdm.autonotebook import trange
import pandas as pd
import os

def get_ckipbert(model_id: str = 'ckiplab/bert-base-chinese', device='cpu'):
    tokenizer = BertTokenizerFast.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    return model, tokenizer

def inference(tokenizer, model, sentences, batch_size=16, verbose=False):
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

    embeddings = []
    time_per_batch = []
    with tqdm(total=len(sentences), desc="Batches", disable=not verbose) as pbar:
        for i in trange(0, len(sentences), batch_size, desc="Batches", disable=True):
            start_time = time.time()
            batch = sentences_sorted[i:i+batch_size]
            encoded_inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
            with torch.no_grad():
                output = model(**encoded_inputs)['last_hidden_state'].detach()
                batch_prototypes = torch.mean(output, dim=1)
                batch_prototypes = torch.nn.functional.normalize(batch_prototypes, p=2, dim=1)
                embeddings.extend(batch_prototypes.to('cpu'))
            time_per_batch.append(time.time() - start_time)
            pbar.update(len(batch))

    embeddings = [embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    embeddings = np.asarray([emb.numpy() for emb in embeddings])

    return embeddings, time_per_batch

if __name__ == "__main__":
    """
    Following code will take csv files under ./items and make embeddings for each item and save them as .npy files.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model and tokenizer...")
    model, tokenizer = get_ckipbert(device=device)
    print("Model and tokenizer loaded successfully.")

    csv_files = [file for file in os.listdir('./random_samples_1M') if file.endswith('.csv')]
    print(f"Found {len(csv_files)} csv files under ./random_samples_1M")

    with tqdm(total=len(csv_files), desc="Inferencing embeddings") as pbar:
        for csv_file in csv_files:
            items_df = pd.read_csv(f'./random_samples_1M/{csv_file}')
            items_df['product_name'] = items_df['product_name'].astype(str)

            product_names = items_df['product_name'].values
            
            embeddings, _ = inference(tokenizer, model, product_names, device=device, batch_size=32, verbose=True)
            embeddings = embeddings.astype(np.float16)
            
            if not os.path.exists('./embeddings'):
                os.makedirs('./embeddings')
            if not os.path.exists(f'./embeddings/ckipbert'):
                os.makedirs(f'./embeddings/ckipbert')
            np.save(f'./embeddings/ckipbert/{csv_file[:-4]}.npy', embeddings)
            
            pbar.update(1)

    print("Embeddings saved successfully.")
