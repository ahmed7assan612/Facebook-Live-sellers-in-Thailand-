from flask import Flask, request, jsonify
import pickle
import numpy as np

with open('kmeans_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    features = np.array(data['status_type','num_reactions','num_comments','num_shares','num_likes','num_loves','num_wows','num_hahas','num_sads','num_angrys'])
    
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    cluster = kmeans.predict(features)
    
    return jsonify({'cluster': int(cluster[0])})

if __name__ == '__main__':
    app.run(debug=True)
