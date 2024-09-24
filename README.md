# Semantic Search System for NTUST Big Data Analysis Course (EE5327701)

This repository provides a semantic search system tailored for the **NTUST Big Data Analysis course (EE5327701)**. It allows users to generate embeddings for e-commerce product descriptions using either a distilled, quantized EcomBERT model or the CKIP BERT model. The system then utilizes FAISS for efficient similarity search, enabling semantic search over large datasets.

*Special thanks to William Wu (clw8998) for creating the distilled and quantized models used in this project.*

## Features

- **Semantic Search:** Perform semantic search over product descriptions using pre-computed embeddings.
- **Embedding Generation:** Optionally generate embeddings using the **EcomBERT** model (`semantic_model`) or the **CKIP BERT** model (`ckipbert`).
- **FAISS Integration:** Build FAISS indexes for efficient similarity search over large datasets.
- **Interactive Search:** Perform interactive semantic search queries through a command-line interface.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Performing Semantic Search](#performing-semantic-search)
  - [Generating Embeddings (Optional)](#generating-embeddings-optional)
- [Scripts Overview](#scripts-overview)
  - [`example_search.py`](#example_searchpy)
  - [`ckipbert.py`](#ckipbertpy)
  - [`semantic_model.py`](#semantic_modelpy)
  - [`get_dataset.py`](#get_datasetpy)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/agbld/semantic-search-for-EE5327701.git
   cd semantic-search-for-EE5327701
   ```

2. **Install Python Dependencies:**
   Ensure you have Python 3.7 or higher installed. Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** The key packages include `transformers`, `sentence-transformers`, `faiss-cpu`, `optimum`, `huggingface_hub`, `numpy`, `pandas`, and `tqdm`.

3. **Download Datasets and Pre-computed Embeddings:**

   **Note:** You do not need to run any script explicitly to download the datasets and embeddings. When you run the `example_search.py` script, it will automatically import and execute `get_dataset.py`, which handles downloading and setting up the necessary files.

## Usage

### Performing Semantic Search

Use the `example_search.py` script to perform semantic search over the pre-computed embeddings.

1. **Run the `example_search.py` script:**
   ```bash
   python example_search.py --model_type semantic_model --top_k 5
   ```

   - `--model_type`: Choose between `semantic_model` (EcomBERT) or `ckipbert` (CKIP BERT).
   - `--top_k`: Specify the number of top results to return.

2. **Interactive Search:**
   After running the script, you can enter queries interactively:
   ```
   Enter query (type "exit" to quit): Your search query here
   ```

   The script will output the top matching product descriptions along with their similarity scores.

**Example:**
```bash
$ python example_search.py --model_type semantic_model --top_k 5
Number of products: 10000
Number of pre-computed embeddings: 10000
FAISS index built with 10000 vectors.
Enter query (type "exit" to quit): Wireless Bluetooth Headphones
Took 0.0023 seconds to search
[Rank 1 | Score: 0.9123] Wireless Over-Ear Headphones with Noise Cancellation
[Rank 2 | Score: 0.8976] Bluetooth Earbuds with Charging Case
[Rank 3 | Score: 0.8765] Noise-Cancelling Wireless Headphones
[Rank 4 | Score: 0.8543] Sports Bluetooth Headset
[Rank 5 | Score: 0.8321] Wireless In-Ear Earphones
```

**Note:** The first time you run the script with a particular model (e.g., `semantic_model` or `ckipbert`), it will automatically download the model from Hugging Face Hub. This may take some time depending on your internet connection speed. Subsequent runs will load the model from the local cache, which will be much faster.

### Generating Embeddings (Optional)

**Note:** Pre-computed embeddings are already provided and downloaded when you run `example_search.py`. Generating embeddings is optional and only necessary if you wish to practice or experiment with the embedding generation process.

You can generate embeddings using either the EcomBERT model (`semantic_model.py`) or the CKIP BERT model (`ckipbert.py`).

#### Using the EcomBERT Model

1. **Run the `semantic_model.py` script:**
   ```bash
   python semantic_model.py
   ```

   This script will:

   - Load the distilled EcomBERT model.
   - Process CSV files under `./random_samples_1M/`.
   - Generate embeddings and save them as `.npy` files under `./embeddings/semantic_model/`.

   **Note:** The first time you run this script, it will automatically download the EcomBERT model from Hugging Face Hub. This may take some time.

#### Using the CKIP BERT Model

1. **Run the `ckipbert.py` script:**
   ```bash
   python ckipbert.py
   ```

   This script will:

   - Load the CKIP BERT model.
   - Process CSV files under `./random_samples_1M/`.
   - Generate embeddings and save them as `.npy` files under `./embeddings/ckipbert/`.

   **Note:** The first time you run this script, it will automatically download the CKIP BERT model from Hugging Face Hub. This may take some time.

## Scripts Overview

### `example_search.py`

- **Purpose:** Perform semantic search using FAISS over the pre-computed embeddings.
- **Features:**
  - **Automatic Dataset Setup:** Imports `get_dataset.py`, which automatically downloads and sets up datasets and embeddings if they are not already present.
  - Loads embeddings and product names.
  - Builds a FAISS index for efficient similarity search.
  - Provides an interactive command-line interface for entering queries.
- **Usage:**
  ```bash
  python example_search.py --model_type semantic_model --top_k 5
  ```

### `ckipbert.py`

- **Purpose:** Generate embeddings using the CKIP BERT model (optional).
- **Functions:**
  - `get_ckipbert(model_id: str, device)`: Loads the CKIP BERT tokenizer and model.
  - `inference(tokenizer, model, sentences, device, batch_size, verbose)`: Generates embeddings for a list of sentences.
- **Usage:** Processes CSV files and saves embeddings under `./embeddings/ckipbert/`.

### `semantic_model.py`

- **Purpose:** Generate embeddings using the distilled, quantized EcomBERT model (optional).
- **Functions:**
  - `get_semantic_model(model_id: str)`: Loads the EcomBERT tokenizer and model.
  - `inference(tokenizer, model, sentences, batch_size, verbose)`: Generates embeddings for a list of sentences.
- **Usage:** Processes CSV files and saves embeddings under `./embeddings/semantic_model/`.

### `get_dataset.py`

- **Purpose:** Download datasets and pre-computed embeddings from Hugging Face Hub.
- **Functionality:**
  - Checks if datasets and embeddings already exist before downloading.
  - Downloads `semantic_model.zip`, `ckipbert.zip`, and `random_samples_1M.zip`.
  - Unzips and organizes files into appropriate directories.
- **Note:** You do not need to run this script explicitly. It is automatically imported and executed when running `example_search.py`.

---

This repository is created for academic purposes as part of the NTUST Big Data Analysis course (EE5327701). Feel free to modify and extend it for other projects!