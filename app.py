from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

def load_model():
    try:
        file_path = os.path.join(os.getcwd(), 'final_model.pkl')
        print(f"Attempting to load model from {file_path}")
        with open('final_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure 'model.pkl' is in the correct directory.")
        return None

model = load_model() 


def validate_input(data):
    ranges = {
        'Adult Mortality': (0, 1000),
        'BMI': (0, 100),
        'HIV/AIDS': (0, 50),
        'Population': (0, 1000000000),
        'Income composition of resources': (0, 1),
        'Schooling': (0, 20)
    }
    errors = []
    for feature, value in data.items():
        if feature in ranges:
            min_val, max_val = ranges[feature]
            if value < min_val or value > max_val:
                errors.append(f"{feature}: Value must be between {min_val} and {max_val}")
    return errors

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        adult_mortality = float(request.form['adult_mortality'])
        bmi = float(request.form['bmi'])
        hiv_aids = float(request.form['hiv_aids'])
        income_composition = float(request.form['income_composition'])
        schooling = float(request.form['schooling'])

        # Prepare the input data for prediction
        input_data = {
            'Adult Mortality': adult_mortality,
            'BMI': bmi,
            'HIV/AIDS': hiv_aids,
            'Income composition of resources': income_composition,
            'Schooling': schooling
        }

        # Validate the input data
        errors = validate_input(input_data)

        if errors:
            return render_template('index.html', errors=errors)

        # Convert the input data to a numpy array
        input_data = np.array(list(input_data.values())).reshape(1, -1)

        # Use the loaded model to make the prediction
        prediction = model.predict(input_data)[0]

        # Redirect to the result page with the prediction
        return redirect(url_for('result', prediction=float(prediction)))

    # If it's a GET request, render the index page
    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
