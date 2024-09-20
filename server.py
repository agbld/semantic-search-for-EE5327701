import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./save_model/ABRSS_student_L3_onnx_QINT8")
parser.add_argument("--model_type", type=str, default="ecombert", choices=["ecombert", "ckipbert"])
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

if args.model_type == "ecombert":
    from ecombert import get_ecombert, inference
    model, tokenizer = get_ecombert(args.model_path)
elif args.model_type == "ckipbert":
    from ckipbert import get_ckipbert, inference
    model, tokenizer = get_ckipbert(args.model_path)

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