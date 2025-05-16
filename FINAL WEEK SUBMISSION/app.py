from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the crop recommendation model
def load_crop_model():
    try:
        nb_model = pickle.load(open('/Users/apple/Desktop/FINAL WEEK PROJECT /naive_bayes_model.pkl', 'rb'))
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    return nb_model

# Crop number to name mapping
crop_dict = {
    1: 'rice',
    2: 'maize',
    3: 'jute',
    4: 'cotton',
    5: 'coconut',
    6: 'papaya',
    7: 'orange',
    8: 'apple',
    9: 'muskmelon',
    10: 'watermelon',
    11: 'grapes',
    12: 'mango',
    13: 'banana',
    14: 'pomegranate',
    15: 'lentil',
    16: 'blackgram',
    17: 'mungbean',
    18: 'mothbeans',
    19: 'pigeonpeas',
    20: 'kidneybeans',
    21: 'chickpea',
    22: 'coffee'
}

nb_model = load_crop_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input values from the form
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Prepare features and predict
            crop_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            crop_prediction_num = nb_model.predict(crop_features)[0]
            crop_name = crop_dict.get(crop_prediction_num, "Unknown Crop")

            return render_template('index.html', crop_prediction=crop_name)
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
