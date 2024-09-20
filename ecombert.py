"""
This file contains the functions to load and infer the distilled, quantized EcomBERT model.
"""

import torch
import time
import numpy as np

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
    
def inference(tokenizer, model, sentences, batch_size):
    
    length_sorted_idx = np.argsort([-_text_length(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

    embeddings = []
    time_per_batch = []
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

    embeddings = [embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    embeddings = np.asarray([emb.numpy() for emb in embeddings])
    

    return embeddings, time_per_batch

def get_ecombert(model_id: str = './save_model/ABRSS_student_L3_onnx_QINT8'):
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = get_ecombert()
    sentences = ['I love you', 'I hate you']
    embeddings, time_per_batch = inference(tokenizer, model, sentences, 2)
    print(embeddings.shape)
    print(time_per_batch)