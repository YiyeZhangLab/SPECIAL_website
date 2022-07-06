from flask import Flask, jsonify, request
import pickle
import shap
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello():
    return "Connection Checked!"

@app.route('/predict', methods=['POST'])
def predict():
	if model:
		json_ = request.json
		query_df = pd.DataFrame(json_, index = [0])
		query = query_df.drop(columns = ['patient_id'])
#		query = pd.get_dummies(query_df)
		prediction = model.predict(query)
		return jsonify(prediction[0])
		
	else:
		print('No model here to use')

@app.route('/shap', methods=['POST'])
def shap():
#	fh = open('explainer.pickle','rb')
#	explainer = joblib.load('explainer.pkl')
	json_ = request.json
	query_df = pd.DataFrame(json_, index = [0])
	query = query_df.drop(columns = ['patient_id'])
	shap_values = explainer.shap_values(query)
	maxindex = np.argmax(shap_values[0])
	maxfeature = query.columns[maxindex]

	return jsonify(maxfeature)



if __name__ == '__main__':
	try:
		port = int(sys.argv[1])
	except:
		port = 12345
	model = joblib.load("clfLogisticRegression.pkl")
	explainer = joblib.load('explainer.pkl')
	print('Model loaded')
#	from waitress import serve
#	serve(app, host= "0.0.0.0", port = 8080)
	app.run( port = port, debug = True)