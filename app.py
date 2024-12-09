from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

pipeline_classification = joblib.load('random_forest_classification_pipeline.pkl')
pipeline_regression = joblib.load('random_forest_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')

expected_columns_classification = ['CUPO_L1', 'Fac_Media', 'CUPO_MX', 'Txs_Media']
expected_columns_regression = ['TC', 'Dualidad', 'Cuentas', 'Antiguedad_Anios', 'CUPO_MX', 'PagoNac_Media', 'Fac_Media']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classification', methods=['GET', 'POST'])
def classification_page():
    if request.method == 'POST':
        try:
            features = [float(request.form.get(col, 0)) for col in expected_columns_classification]
            features_array = np.array(features).reshape(1, -1)
            prediction = pipeline_classification.predict(features_array)
            prediction_label = label_encoder.inverse_transform(prediction)[0]
            return render_template('classification.html', prediction=prediction_label, success=True)
        except Exception as e:
            return render_template('classification.html', error=str(e))
    return render_template('classification.html')

@app.route('/regression', methods=['GET', 'POST'])
def regression_page():
    if request.method == 'POST':
        try:
            features = [float(request.form.get(col, 0)) for col in expected_columns_regression]
            features_array = np.array(features).reshape(1, -1)
            prediction = pipeline_regression.predict(features_array)
            prediction_list = prediction.tolist()
            return render_template('regression.html', prediction=prediction_list[0], success=True)
        except Exception as e:
            return render_template('regression.html', error=str(e))
    return render_template('regression.html')

if __name__ == '__main__':
    app.run(debug=True)
