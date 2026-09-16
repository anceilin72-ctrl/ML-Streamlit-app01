import streamlit as st
import joblib

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸"
)

# Load the trained model
model = joblib.load("iris_model.pkl")

flower_names = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

st.title("Iris Flower Prediction")

st.write(
    "Enter the flower measurements to predict "
    "its Iris species."
)

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1
)

if st.button("Predict"):
    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    predicted_flower = flower_names[prediction]
    confidence = probabilities[prediction] * 100

    st.success(
        f"Predicted Iris class: {predicted_flower}"
    )

    st.info(
        f"Prediction confidence: {confidence:.2f}%"
    )