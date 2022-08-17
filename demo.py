from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

import nmslib

n = 100
data = np.random.rand(10000, n).astype(np.float32)
index = nmslib.init(method='hnsw', space='l2')
index.addDataPointBatch(data)
index.createIndex(print_progress=True)

app = Flask(__name__)

@app.route('/')
def default():
    return "recommandation system demo"

@app.route('/api/rec', methods=['GET', 'POST'])
def api_rec():
    args = request.args
    ids = []
    if request.method == 'GET':
        print('GET')
        try:
            k = int(args.get('k'))
            id = int(args.get('id'))
            ids, distance = index.knnQuery(data[id], k=k)
        except:
            pass
    if request.method == 'POST':
        try:
            query_vector = request.json['vector']
            print(query_vector, type(query_vector), type(query_vector[0]))
            k = int(request.json['k'])
            ids, distance = index.knnQuery(query_vector, k=k)
        except:
            pass

    if len(ids) == 0:
        return "invalid query", 500
    return jsonify(ids.tolist()), 200


app.run(host='0.0.0.0', port=81)
