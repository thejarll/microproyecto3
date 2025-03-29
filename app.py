from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo
modelo = joblib.load("modelo_cafe_simple.pkl")

# Columnas usadas
feature_names = [
    "Number_of_Customers_Per_Day",
    "Average_Order_Value",
    "Marketing_Spend_Per_Day"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]
    prediction = modelo.predict(df)[0]
    return jsonify({"predicted_revenue": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
