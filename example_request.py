import requests

def get_embeddings(text: list, url: str = 'http://localhost:5000/api/embed') -> list:
    url = 'http://localhost:5000/api/embed'
    headers = {'Content-Type': 'application/json'}
    data = {'text': text}

    response = requests.post(url, json=data, headers=headers)
    return response.json()

data = ["Ding Dong Pet 寵物貓 Sumsum 洞穴式隧道屋, 考拉, 1個", 
        "dingdog 小狗狗狗鼻子走道零食玩具, 茄子, 1個", 
        "TUFFY 拉扯拔河玩具 經典基本款耐咬圈圈, 紅磚, 1個", 
        "Celebpet Latex Sweet Macaron Beep Dog Toy 5 x 5 x 3 cm, 隨機發貨, 3個",
        "YOGiSSO 魚造型貓薄荷玩偶, 黑色, 1入",]

embeddings = get_embeddings(data)

num_embeddings = len(embeddings)
num_dimension = len(embeddings[0])
print(f"Number of embeddings: {num_embeddings}")
print(f"Dimension of embeddings: {num_dimension}")