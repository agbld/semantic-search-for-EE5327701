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

2. **Make Sure You Have PyTorch Installed:**
   Ensure you have PyTorch installed. If not, you can install it from the [official website](https://pytorch.org/).

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

**Note:** The first time you run the script with a particular model (e.g., `semantic_model` or `ckipbert`), it will automatically download the model from Hugging Face Hub. This may take some time depending on your internet connection speed. Subsequent runs will load the model from the local cache, which will be much faster.

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

EcomBERT Model:
```bash
$ python example_search.py 
Number of products: 1000000
Number of pre-computed embeddings: 1000000
FAISS index built with 1000000 vectors.
Enter query (type "exit" to quit): Wireless Bluetooth Headphones
Took 0.1734 seconds to search
[Rank 1 | Score: 0.7561] Wireless 2矽膠耳機殼+扣, 單品, 白色（案例）
[Rank 2 | Score: 0.7478] NS_AIR4 WIRELESS EARBUDS 藍芽耳機(黑), 1個
[Rank 3 | Score: 0.7199] Kinyo 藍牙耳機 60 x 27 x 53mm 充電盒36g 單支耳機4g, BTE-3905, 1個
[Rank 4 | Score: 0.7138] 登山扣修身耳機盒, 海軍, 谷歌像素芽 2
[Rank 5 | Score: 0.7121] DEKONI AUDIO Deco -Bluetooth耳機耳朵尖端TWS泡沫提示6p, 交易平台_M, 單色
Enter query (type "exit" to quit): 寵物玩具
Took 0.1423 seconds to search
[Rank 1 | Score: 0.9329] 寵物狗玩具 3入 S號, 混色, 1套
[Rank 2 | Score: 0.9323] 動物造型寵物玩具組 3入, 隨機發貨, 1套
[Rank 3 | Score: 0.9270] SUPER PET 寵物用玩具組, 隨機發貨（紫薯）, 1套
[Rank 4 | Score: 0.9227] multipet 絨毛寵物玩具 L號, 1個, 隨機發貨
[Rank 5 | Score: 0.9208] 青年商城寵物玩具耐用4件套, 混色, 1組
Enter query (type "exit" to quit): 洗衣精
Took 0.1622 seconds to search
[Rank 1 | Score: 0.8855] 茶樹莊園 超濃縮洗衣精補充包 天然抗菌, 1.5kg, 4包
[Rank 2 | Score: 0.8850] 茶樹莊園 超濃縮洗衣精 純淨消臭, 1.8kg, 5瓶
[Rank 3 | Score: 0.8834] 茶樹莊園 茶樹洗衣精組合包, 茶樹洗衣精2000g+茶樹洗衣精補充包1500g, 1組
[Rank 4 | Score: 0.8829] 茶樹莊園 超濃縮洗衣精 純淨消臭, 1.8kg, 3瓶
[Rank 5 | Score: 0.8755] 茶樹莊園 超濃縮洗衣精補充包 天然抗菌, 1.5kg, 3包
```

CKIP BERT Model:
```bash
$ python example_search.py --model_type ckipbert
Number of products: 1000000
Number of pre-computed embeddings: 1000000
FAISS index built with 1000000 vectors.
Enter query (type "exit" to quit): Wireless Bluetooth Headphones
Took 0.1463 seconds to search
[Rank 1 | Score: 0.8343] Foot-On Jaguar Fine Pattern 消聲器
[Rank 2 | Score: 0.8301] VRS Dewallet Hybrid Origin MagSafe 卡儲存支架可拆卸手機殼
[Rank 3 | Score: 0.8263] EXPEAK Tracking Climbing 休閒智能手機袋 黃色
[Rank 4 | Score: 0.8239] LEADCOOL ARGB記憶體散熱器 4入, RH-1 EVO
[Rank 5 | Score: 0.8236] Rykel Allround Grip 2 磁性支架
Enter query (type "exit" to quit): 寵物玩具
Took 0.1451 seconds to search
[Rank 1 | Score: 0.8563] 寵物餵食器玩具, 白色的
[Rank 2 | Score: 0.8274] jw 寵物活動玩具四足小鳥玩具, 1個
[Rank 3 | Score: 0.8230] 寵物睡墊, 綠色
[Rank 4 | Score: 0.8188] 動物造型變形機器人玩具, 狼
[Rank 5 | Score: 0.8184] 動物造型變形機器人玩具, 豹
Enter query (type "exit" to quit): 洗衣精
Took 0.1454 seconds to search
[Rank 1 | Score: 0.7846] 洗髮精 直接擦鞋劑
[Rank 2 | Score: 0.7517] 洗衣烘乾架, 1個
[Rank 3 | Score: 0.7513] 洗衣劑, 2個, 1.7L
[Rank 4 | Score: 0.7503] 洗衣機防塵罩, 5
[Rank 5 | Score: 0.7491] 洗碗機用液體洗滌劑, 1L, 1個
```

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