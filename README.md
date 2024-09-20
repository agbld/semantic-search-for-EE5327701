# EcomBERT Embedding System for NTUST Big Data Analysis Course (EE5327701)

This repository contains a system that uses a distilled and quantized EcomBERT model to generate embeddings for e-commerce product descriptions. It is designed as an demonstration project for the **NTUST Big Data Analysis course (EE5327701)**.

*Special thanks to William Wu (clw8998), who create the distillation and quantization model for this project. It is actually a fantastic work!*

## Features
- **Efficient Embedding Generation:** Utilizes a distilled and quantized version of EcomBERT for fast inference and reduced model size.
- **Batch Processing:** Handles multiple input sentences in batches, leveraging ONNX quantization for performance.
- **API Integration:** Provides a RESTful API endpoint for generating embeddings using Flask (handled by the TA).
- **Multilingual Support:** Capable of handling English and Chinese product descriptions.

## Roles and Responsibilities

### Students
- **Usage:** Students are only responsible for sending requests to the API and do **not** need to host the model API server locally.
- **Focus:** The focus for students is to work on tasks such as retrieving embeddings for product descriptions and performing further analysis or tasks based on the embeddings provided by the API.
- **No need for model files:** Students do not need to download or manage the model files, as the server is hosted by the TA.

### TAs
- **Hosting the API Server:** TAs are responsible for hosting and maintaining the model API server that the students will interact with.
- **Model Management:** TAs will ensure that the pre-trained EcomBERT model is available on the server and correctly set up for inference.

## Table of Contents
- [Installation (For TAs Only)](#installation-for-tas-only)
- [Installation (For Students)](#installation-for-students)
- [Usage](#usage)
- [API Documentation](#api-documentation)

## Installation (For TAs Only)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ecombert-for-EE5327701.git
   cd ecombert-for-EE5327701
   ```

2. **Install PyTorch:**
   If PyTorch is not installed, follow the instructions on the [official website](https://pytorch.org/get-started/locally/) to install the appropriate version based on your system configuration.

   Here's an example for installing PyTorch with CPU-only support:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

   In this repository, we are not neccessary to utilize GPU for the model inference since the model is quantized and optimized for CPU inference. (*again, thanks to William*)

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the EcomBERT model is available in the `./save_model/ABRSS_student_L3_onnx_QINT8` directory.**
   (The model is not included in this repository and it is not necessary for students to download it.)

5. **Start the server:**
   ```bash
   python server.py
   ```

## Installation (For Students)

Students do not need to perform any setup. The API server, managed by the TA, will handle all inference tasks. The only requirement is to send requests to the provided API endpoint.

## Usage

The main task for students is to retrieve embeddings by interacting with the API server. Below is an example of how students can send requests to the API:

1. **Send a POST request to the API to get embeddings:**
   Using the provided `example_request.py` script:
   ```bash
   python example_request.py
   ```

   Alternatively, use `curl` to send a request:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"text": ["I love you", "I hate you"]}' http://<TA-server-address>:5000/api/embed
   ```

### Example Product Descriptions

Students can input product descriptions (in English or Chinese) to retrieve embeddings, e.g.:
- "Ding Dong Pet 寵物貓 Sumsum 洞穴式隧道屋, 考拉, 1個"
- "YOGiSSO 魚造型貓薄荷玩偶, 黑色, 1入"

These descriptions will be processed by the API to return vector embeddings.

## API Documentation

The API server exposes a single endpoint:

### `/api/embed` (POST)
This endpoint accepts a list of sentences and returns the corresponding embeddings generated by the EcomBERT model.

#### Request:
- **URL:** `http://<TA-server-address>:5000/api/embed`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Body:** A JSON object containing a list of sentences under the `text` key.
  ```json
  {
    "text": [
      "Ding Dong Pet 寵物貓 Sumsum 洞穴式隧道屋, 考拉, 1個", 
      "TUFFY 拉扯拔河玩具 經典基本款耐咬圈圈, 紅磚, 1個"
    ]
  }
  ```

#### Response:
- A JSON object containing the embeddings for each sentence as a list of floats.
  ```json
  [
    [0.1, 0.2, 0.3, ..., 0.128], 
    [0.5, 0.6, 0.7, ..., 0.256]
  ]
  ```

### Example Usages

1. **Send a request via Python:**
   ```python
   import requests

   data = ["This is an example sentence", "Another test sentence"]
   response = requests.post('http://<TA-server-address>:5000/api/embed', json={'text': data})
   print(response.json())
   ```

2. **Using `curl` for quick testing:**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"text": ["sample product description", "another description"]}' http://<TA-server-address>:5000/api/embed
   ```

## Notes

- **For Students:** You are not required to download the model or set up the API server. The embeddings will be provided by the API hosted by the TA.
- **For TAs:** Make sure the API server is up and running throughout the lab sessions, and ensure the model is loaded properly for generating embeddings.
- This project is designed for academic purposes as part of the NTUST Big Data Analysis course (EE5327701). Feel free to extend the system for other applications, such as product recommendation or semantic search.

---

This repository is created for academic purposes as part of the NTUST Big Data Analysis course (EE5327701). Students and TAs are encouraged to collaborate for the successful completion of this project!