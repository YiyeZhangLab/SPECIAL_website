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
		prediction = model.predict_proba(query)
#		print(prediction)
		return jsonify(prediction[0][1])
		
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
#	shap_max_values = np.absolute(shap_values)
#	maxindex = np.argmax(shap_values[0])
	sorted_shap_indices = np.argsort(shap_values[0])[::-1]
	top_3_shap_indices = sorted_shap_indices[:3]
#	maxindex_abs = np.argmax(shap_max_values[0])
	top_3_features = query.columns[top_3_shap_indices]
#	maxfeature = query.columns[maxindex]
#	maxfeature_abs = query.columns[maxindex_abs]
#	print(shap_max_values)
#	print(maxfeature_abs)

	return jsonify(list(top_3_features))



if __name__ == '__main__':
	try:
		port = int(sys.argv[1])
	except:
		port = 12345
	model = joblib.load("clfLogisticRegression.pkl")
	explainer = joblib.load('explainer_ver202207.pkl')
	print('Model loaded')
#	from waitress import serve
#	serve(app, host= "0.0.0.0", port = 8080)
	app.run( port = port, debug = True)
