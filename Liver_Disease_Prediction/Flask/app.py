# Import necessary libraries
from flask import Flask, render_template, request
from joblib import load
import numpy as np
import warnings


# Initialize the Flask application
app = Flask(__name__)

# Define the home page route
@app.route('/')
def home():
    return render_template("home.html")

# Define the route to show the input form
@app.route('/predict')
def index():
    return render_template("index.html")

# Define the route to handle the form submission
@app.route('/data_predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = request.form['age']
    gender = request.form['gender']
    tb = request.form['tb']
    db = request.form['db']
    ap = request.form['ap']
    aa1 = request.form['aa1']
    aa2 = request.form['aa2']
    tp = request.form['tp']
    a = request.form['a']
    agr = request.form['agr']

    # Create a list with the input values
    data = [[float(age), float(gender), float(tb), float(db), float(ap), float(aa1), float(aa2), float(tp), float(a), float(agr)]]

    # Load the trained model
    model = load("C:/Users/ELCOT/Desktop/ARun project/ETC.joblib")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Make a prediction using the model
        prediction = model.predict(np.array(data))[0]


    # Show the appropriate message based on the prediction result
        if prediction == 1:
            return render_template('noChance.html', prediction='You have a liver disease problem. You must consult a doctor.')
        else:
            return render_template('chance.html', prediction='You do not have a liver disease problem.')
# Run the Flask application
if __name__ == '__main__':
    app.run()
