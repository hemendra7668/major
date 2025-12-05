import streamlit as st
import pandas as pd, joblib, numpy as np, os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vitals_multioutput_rf.pkl")
model = joblib.load(MODEL_PATH)

st.title("Patient Vitals — Dashboard")
st.write("Upload a CSV with columns: Age,Gender,BMI,Derived_HRV,Derived_Pulse_Pressure,Derived_MAP to get batch predictions.")

uploaded = st.file_uploader("CSV file", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())
    features = ["Age","Gender","BMI","Derived_HRV","Derived_Pulse_Pressure","Derived_MAP"]
    X = df[features].fillna(0)
    preds = model.predict(X)
    preds_df = pd.DataFrame(preds, columns=["Heart Rate","Respiratory Rate","Body Temperature","Oxygen Saturation","Systolic BP","Diastolic BP"])
    out = pd.concat([df.reset_index(drop=True), preds_df], axis=1)
    st.write("Predictions:", out.head())
    st.download_button("Download predictions CSV", out.to_csv(index=False), file_name="predictions.csv")
else:
    st.write("No CSV uploaded. Try example:")
    sample = pd.DataFrame([{"Age":45,"Gender":0,"BMI":24.5,"Derived_HRV":5,"Derived_Pulse_Pressure":40,"Derived_MAP":90}])
    st.write(sample)
    if st.button("Predict sample"):
        p = model.predict(sample)[0]
        st.write({"Heart Rate":p[0],"Respiratory Rate":p[1],"Body Temperature":p[2],"Oxygen Saturation":p[3],"Systolic BP":p[4],"Diastolic BP":p[5]})
