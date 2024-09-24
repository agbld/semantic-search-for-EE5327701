You're right, the Installation section is redundant given the updated steps for students. Here's the revised README.md without the Installation section and with the TA parts removed:

---

# Semantic Searching System for NTUST Big Data Analysis Course (EE5327701)

This repository contains a system that generates embeddings for e-commerce product descriptions using both the distilled and quantized **Semantic model** and the **CKIP BERT** model. It is designed as a demonstration project for the **NTUST Big Data Analysis course (EE5327701)**.

*Special thanks to William Wu (clw8998), who created the distillation and quantization model for this project. It's truly outstanding work!*

## Features

- **Dual Model Support:** Provides options to use either the Semantic model or the CKIP BERT model for generating embeddings.
- **Efficient Embedding Generation:** Utilizes a distilled and quantized version of the Semantic model and the optimized CKIP BERT for fast inference and reduced model size.
- **Batch Processing:** Handles multiple input sentences in batches, leveraging ONNX quantization for performance.
- **API Integration:** Offers a RESTful API endpoint for generating embeddings using Flask.
- **Multilingual Support:** Capable of handling English and Chinese product descriptions.

## Usage

The main tasks are to generate product data and embeddings, start the API server, and perform semantic search using the provided scripts.

### Step 1: Generate Product Data and Embeddings

Run the following command to obtain product names and precomputed embeddings:

```bash
python get_dataset.py
```

This script will:

- Download the product data and save it in the `./items/` directory.
- Generate embeddings for the product descriptions using both the Semantic model and CKIP BERT model.
- Save the embeddings in the `./embeddings/semantic_model/` and `./embeddings/ckipbert/` directories respectively.

### Step 2: Start the API Server

Run the following command to start the Flask API server:

```bash
python server.py
```

This will start the server on `http://localhost:5000`, providing an API endpoint for generating embeddings.

### Step 3: Perform Semantic Search

Use the `example.py` script to perform semantic search.

#### Using Semantic Model

```bash
python example.py --model_type=semantic_model
```

#### Using CKIP BERT

```bash
python example.py --model_type=ckipbert
```

**Example Usage:**

Run the script and enter a query when prompted:

```bash
Enter query: YOGiSSO 魚造型貓薄荷玩偶, 黑色, 1入
```

The script will output the top-k similar products:

```
Takes 0.1234 seconds to search
[Rank 1 (0.95)] YOGiSSO 魚造型貓薄荷玩偶, 黑色, 1入
[Rank 2 (0.85)] Other similar product name
...
```

## API Documentation

### Endpoint

- **URL:** `http://localhost:5000/api/embed`
- **Method:** POST
- **Content-Type:** application/json

### Request Format

- **Body:**

  ```json
  {
    "text": [
      "Product description 1",
      "Product description 2",
      "..."
    ]
  }
  ```

- **Parameters:**

  - `text`: A list of product descriptions (strings) you want to generate embeddings for.

### Response Format

- **Body:**

  ```json
  [
    [0.1, 0.2, 0.3, ..., 0.128],
    [0.5, 0.6, 0.7, ..., 0.256],
    ...
  ]
  ```

- **Details:**

  - Returns a list of embeddings corresponding to each input text.
  - Each embedding is a list of floating-point numbers (vector).

### Example Usage

Using `curl` for quick testing:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": ["Product description 1", "Product description 2"]}' http://localhost:5000/api/embed
```

## Notes

- Ensure that the server is running before attempting to use the API or perform semantic search.
- Use the `--model_type` argument in `example.py` to switch between the Semantic model and CKIP BERT model.
- The `get_dataset.py` script must be run first to ensure all data and embeddings are properly set up.
- Feel free to modify and extend the scripts for your analysis.

---

This repository is created for academic purposes as part of the NTUST Big Data Analysis course (EE5327701). Feel free to modify and extend it for other projects!