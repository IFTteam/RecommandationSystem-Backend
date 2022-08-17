import requests
import numpy as np

url = 'http://localhost:81/api/rec'
myobj = {'k': 10, 'vector': np.random.rand(1, 100).flatten().tolist()}

x = requests.post(url, json = myobj)

print(x.text)