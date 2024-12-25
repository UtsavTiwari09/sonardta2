from flask import Flask, request, jsonify, send_file
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

app = Flask(__name__)

# Load the dataset
sonar_data = pd.read_csv('C:\Users\dell\OneDrive\Desktop\Rock&mine\copy.csv', header=None)

# Prepare the data and model
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60].map({'R': 0, 'M': 1})

# Train the best model (Random Forest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_scaled, Y)

@app.route('/')
def home():
    return 'Welcome to the Sonar Rock or Mine Prediction API'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = list(map(float, data['features']))

        if len(features) != 60:
            return jsonify({"error": "Please provide exactly 60 features."})

        # Preprocess the input features and make a prediction
        input_data = np.array(features).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        prediction_label = 'Rock' if prediction[0] == 0 else 'Mine'

        # Create the confusion matrix for display
        Y_pred = model.predict(X_scaled)
        conf_matrix = confusion_matrix(Y, Y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Rock", "Mine"])

        # Save the confusion matrix image
        cm_display.plot(cmap='Blues')
        cm_img_path = 'confusion_matrix.png'
        plt.savefig(cm_img_path)

        return jsonify({
            'prediction': prediction_label,
            'confusion_matrix_img': cm_img_path
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
