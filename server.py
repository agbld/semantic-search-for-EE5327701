import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./save_model")
parser.add_argument("--model_type", type=str, default="semantic_model", choices=["semantic_model", "ckipbert"])
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

if args.model_type == "semantic_model":
    from semantic_model import get_semantic_model, inference
    model, tokenizer = get_semantic_model(os.path.join(args.model_path, args.model_type))
elif args.model_type == "ckipbert":
    from ckipbert import get_ckipbert, inference
    model, tokenizer = get_ckipbert()

from flask import Flask, request
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=1)

@app.route('/api/embed', methods=['POST'])
def process_string():
    input_strings = request.get_json().get('text')

    def process_request(strings):
        embeddings_list, _ = inference(tokenizer, model, strings , 16)
        return embeddings_list.tolist()

    future = executor.submit(process_request, input_strings)
    return future.result()

if __name__ == '__main__':
    app.run(port=5000, debug=False, threaded=True)

# curl -X POST -H "Content-Type: application/json" -d '{"text": ["hello world", "this is a test", "this is another test", "this might be the last test", "finally"]}' http://localhost:5000/api/embed