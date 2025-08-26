from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('polynomial_regression_model.h5',compile=False)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        x = float(data['x'])  # Expecting JSON input like {"x": 5.0}
        
        # Prepare input for the model (shape: [1, 1] for single input)
        x_input = np.array([[x]])
        
        # Make prediction
        y_pred = model.predict(x_input)[0][0]
        
        # Return prediction as JSON
        return jsonify({'x': x, 'y_pred': float(y_pred)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)