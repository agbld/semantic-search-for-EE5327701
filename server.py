from ecombert import get_ecombert, inference

model, tokenizer = get_ecombert()

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