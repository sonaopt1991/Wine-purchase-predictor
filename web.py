from flask import Flask,render_template,request
import pickle
import numpy as np
app = Flask(__name__)

# Load the trained model
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    """Render the homepage with the input form."""
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction logic."""
    try:
        # Get user input from the form and convert to a list of floats
        features = [float(x) for x in request.form.values()]
        
        # Make a prediction using the model
        prediction = model.predict([np.array(features)])
        
        # Convert prediction to human-readable output
        output = "Will Purchase" if prediction[0] == 1 else "Will Not Purchase"
        

        return render_template('result.html', prediction_text=f"Prediction: {output}")
    
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
