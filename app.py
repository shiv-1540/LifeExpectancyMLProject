from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

def load_model():
    try:
        file_path = os.path.join(os.getcwd(), 'model.pkl')
        print(f"Attempting to load model from {file_path}")
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure 'model.pkl' is in the correct directory.")
        return None

model = load_model() 


def validate_input(data):
    ranges = {
        'Year': (1980, 2025),
        'Status': (0, 1),
        'Alcohol': (0, 20),
        'Adult Mortality': (0, 2000),
        'Hepatitis B': (0, 100),
        'Measles': (0, 100000),
        'BMI': (0, 100),
        'under-five deaths': (0, 1000),
        'Polio': (0, 100),
        'Total expenditure': (0, 20),
        'Diphtheria': (0, 100),
        'HIV/AIDS': (0, 50),
        'GDP': (0, 50000),
        'Population': (0, 1000000000),
        'thinness 1-19 years': (0, 50),
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
        year = int(request.form['year'])
        status = int(request.form['status'])
        alcohol = float(request.form['alcohol'])
        adult_mortality = float(request.form['adult_mortality'])
        hepatitis_b = float(request.form['hepatitis_b'])
        measles = int(request.form['measles'])
        bmi = float(request.form['bmi'])
        under_five_deaths = int(request.form['under_five_deaths'])
        polio = float(request.form['polio'])
        total_expenditure = float(request.form['total_expenditure'])
        diphtheria = float(request.form['diphtheria'])
        hiv_aids = float(request.form['hiv_aids'])
        gdp = float(request.form['gdp'])
        population = float(request.form['population'])
        thinness_1_19_years = float(request.form['thinness_1_19_years'])
        income_composition = float(request.form['income_composition'])
        schooling = float(request.form['schooling'])

        # Prepare the input data for prediction
        input_data = {
            'Year': year,
            'Status': status,
            'Alcohol': alcohol,
            'Adult Mortality': adult_mortality,
            'Hepatitis B': hepatitis_b,
            'Measles': measles,
            'BMI': bmi,
            'under-five deaths': under_five_deaths,
            'Polio': polio,
            'Total expenditure': total_expenditure,
            'Diphtheria': diphtheria,
            'HIV/AIDS': hiv_aids,
            'GDP': gdp,
            'Population': population,
            'thinness 1-19 years': thinness_1_19_years,
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

if __name__ == '__main__':
    app.run(debug=True)