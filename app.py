import streamlit as st
import pickle
import json
import numpy as np

# Page styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
            padding: 20px;
            border-radius: 10px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .stButton>button {
            color: white;
            background-color: #2e86de;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 Hospital Readmission Predictor")

# Load model and features
model = pickle.load(open("model.pkl", "rb"))
model_columns = json.load(open("model_columns.json"))

# Reset functionality
if st.button("🔄 Reset All Fields"):
    st.session_state["age_label"] = "[70-80)"
    st.session_state["time_in_hospital"] = 1
    st.session_state["number_inpatient"] = 0
    st.session_state["number_emergency"] = 0
    st.session_state["number_outpatient"] = 0
    st.rerun()

st.markdown("### 📋 Enter Patient Information:")

# Inputs
age_label = st.selectbox("🧓 Age", ["[70-80)", "[60-70)", "[50-60)", "[40-50)", "[30-40)"], key="age_label")
age_map = {"[70-80)": 0, "[60-70)": 1, "[50-60)": 2, "[40-50)": 3, "[30-40)": 4}
age = age_map[age_label]

time_in_hospital = st.slider("🏥 Time in Hospital (days)", 1, 14, key="time_in_hospital")
number_inpatient = st.number_input("🛌 Inpatient Visits", 0, 20, key="number_inpatient")
number_emergency = st.number_input("🚨 Emergency Visits", 0, 20, key="number_emergency")
number_outpatient = st.number_input("🏃 Outpatient Visits", 0, 20, key="number_outpatient")

# Feature Engineering
total_visits = number_inpatient + number_emergency + number_outpatient
st.write(f"🧾 **Total Prior Visits**: `{total_visits}`")

# Prediction input
input_dict = {
    'age': age,
    'time_in_hospital': time_in_hospital,
    'number_inpatient': number_inpatient,
    'number_emergency': number_emergency,
    'number_outpatient': number_outpatient,
    'total_visits': total_visits
}
input_vector = [input_dict[col] for col in model_columns]

# --- Prediction block ---
if st.button("🔍 Predict Readmission"):
    try:
        prediction = model.predict(np.array(input_vector).reshape(1, -1))[0]
        labels = ["Not Readmitted", "Readmitted within 30 days"]
        values = [1, 0] if prediction == 1 else [0, 1]

        # Show prediction
        if prediction == 1:
            st.success("✅ Readmitted within 30 days")
        else:
            st.error("❎ Not Readmitted")

        # --- Bar chart ---
        st.markdown("### 📊 Prediction Visualization")
        st.bar_chart({
            "Prediction": {
                "Not Readmitted": values[1],
                "Readmitted": values[0]
            }
        })

    except Exception as e:
        st.warning(f"⚠️ Error during prediction: {str(e)}")

