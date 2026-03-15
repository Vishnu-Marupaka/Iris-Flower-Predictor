from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    
    if request.method == 'POST':
        try:
            # Get values from the HTML form
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            # Create a DataFrame with the same column names as training
            data = pd.DataFrame({
                'sepal_length': [sepal_length],
                'sepal_width': [sepal_width],
                'petal_length': [petal_length],
                'petal_width': [petal_width]
            })

            # Predict
            prediction = model.predict(data)[0]
            prediction_text = f"The flower is: {prediction}"
            
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', result=prediction_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)