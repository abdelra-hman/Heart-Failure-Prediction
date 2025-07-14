import gradio as gr
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_heart_failure(age, anaemia, creatinine_phosphokinase, diabetes,
                          ejection_fraction, high_blood_pressure, platelets,
                          serum_creatinine, serum_sodium, sex, smoking, time):

    platelets_per_age = platelets / (age + 1)
    creatinine_per_ck = serum_creatinine / (creatinine_phosphokinase + 1)

    X = np.array([[age, anaemia, creatinine_phosphokinase, diabetes,
                   ejection_fraction, high_blood_pressure, platelets,
                   serum_creatinine, serum_sodium, sex, smoking, time,
                   platelets_per_age, creatinine_per_ck]])

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return "✅ Survived" if prediction == 0 else "❌ Death Expected"

iface = gr.Interface(
    fn=predict_heart_failure,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio([0, 1], label="Anaemia"),
        gr.Number(label="Creatinine Phosphokinase"),
        gr.Radio([0, 1], label="Diabetes"),
        gr.Number(label="Ejection Fraction"),
        gr.Radio([0, 1], label="High Blood Pressure"),
        gr.Number(label="Platelets"),
        gr.Number(label="Serum Creatinine"),
        gr.Number(label="Serum Sodium"),
        gr.Radio([0, 1], label="Sex"),
        gr.Radio([0, 1], label="Smoking"),
        gr.Number(label="Time"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🫀 Heart Failure Predictor",
    description="Enter clinical data to predict heart failure outcome.",
)

iface.launch()
