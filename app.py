from flask import Flask,request,render_template
import numpy as np
#import pandas
import sklearn
import pickle

# Importar los modelos
model = pickle.load(open('model_segmentation.pkl','rb'))
sc = pickle.load(open('standscaler_segmentation.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))

# crear flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    G = request.form['Gender']
    A = request.form['Age']
    Ai = request.form['Anual Income']
    SS = request.form['Spending Score']

    #feature_list = [G, A, Ai, SS]
    #single_pred = np.array(feature_list).reshape(1, -1)

    def prediction(G,A,Ai,SS):
        features = ([[G,A,Ai,SS]])
        transformed_features = encoder.transform(features)
        transformed_features[:,2:] = sc.transform(transformed_features[:,2:])
        prediction = model.predict(transformed_features).reshape(1,-1)
        return prediction[0]
    
    prediction = prediction(G, A, Ai, SS)

    diccionario = {cluster: f'Cluster {cluster}' for cluster in range(0, 5)}

    if prediction[0] in diccionario:
        crop = diccionario[prediction[0]]
        result =("{}".format(crop))
    else:
        result =("Sorry, tas tilin")
    return render_template('index.html',result = result)




# python main
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)