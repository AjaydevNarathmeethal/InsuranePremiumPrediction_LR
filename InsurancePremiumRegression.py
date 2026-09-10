import joblib
import streamlit as st
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("InsurancePremiumPrediction.pkl")

linear_model = load_model()


st.title("🪙 Insurance Premium Predictor")
st.write("Enter the details below to get a real-time prediction of Insurance Premium.")

st.subheader("Enter your details")

# first row
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    age = st.number_input(
        "Age (years)", min_value=18, max_value=85, value=18, step=1
    )

with row1_col2:
    children = st.number_input(
        "Number of Children", min_value=0, max_value=10, value=0, step=1
    )


# second row
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    smoker_input = st.selectbox("Is the individual a smoker?", ["NO", "YES"])

with row2_col2:
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
    )


region_input = st.selectbox(
    "Region of residence", ["Northeast", "Northwest", "Southeast", "Southwest"]
)



if st.button("Calculate insurance Premium"):

    # 1. Map Smoker (Only need 'smoker_yes')
    # If YES -> 1, If NO -> 0
    smoker_yes = 1.0 if smoker_input == "YES" else 0.0

    # 2. Map Region (northeast is the baseline/dropped column)
    region_northwest = 1 if region_input == "Northwest" else 0
    region_southeast = 1 if region_input == "Southeast" else 0
    region_southwest = 1 if region_input == "Southwest" else 0

    # 3. Combine into the EXACT order your model was trained on
    input_features = np.array(
        [
            [
                age,
                bmi,
                children,
                smoker_yes,
                region_northwest,
                region_southeast,
                region_southwest,
            ]
        ]
    )
    print("done")
    # 4. Predict
    prediction = linear_model.predict(input_features)
    predicted_value = prediction[0]
    # 5. Display Result
    st.success(f"💰 The predicted charges are: **${predicted_value:.2f}**")