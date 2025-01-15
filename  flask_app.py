from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load('final_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON input and convert to DataFrame
    input_data = request.json
    df = pd.DataFrame(input_data)

    # Ensure columns match the training features
    df = df[model.feature_names_in_]

    # Make predictions
    predictions = model.predict(df)

    # Return predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)