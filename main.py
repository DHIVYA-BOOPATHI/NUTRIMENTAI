import streamlit as st
import cv2
import numpy as np
from PIL import Image
from roboflow import Roboflow
import tempfile
import os
from py_edamam import Edamam
from bmi import calculate_bmr, calculate_tdee

# Store these values in Streamlit: Manage app -> Settings -> Secrets
rf = Roboflow(api_key=st.secrets["ROBOFLOW_API_KEY"])
workspace = rf.workspace("dhivyas-workspace-i4ksn")
project = workspace.project("food-detection-2eqch-mmfp4")
model = project.version(1).model


def detect_image(image_path):
    """Return a detected food class, or None if no predictions are found."""
    result = model.predict(image_path, confidence=30, overlap=30).json()
    predictions = result.get("predictions") or []

    if not predictions:
        return None

    return predictions[0].get("class") or None


def getcalories(food_item):
    e = Edamam(
        nutrition_appid=st.secrets["EDAMAM_APP_ID"],
        nutrition_appkey=st.secrets["EDAMAM_APP_KEY"],
    )
    data = e.search_nutrient("1 " + str(food_item))

    if not data:
        raise ValueError("No nutrition data was returned.")

    return data.get("calories"), data.get("dietLabels") or [], data.get("totalWeight")


def main():
    st.title("Nutriment AI")
    st.write("Please enter the following information:")

    height = st.number_input(
        "Height (m)", min_value=0.1, value=1.7, step=0.1, format="%.1f"
    )
    weight = st.number_input(
        "Weight (kg)", min_value=1.0, value=60.0, step=0.1, format="%.1f"
    )
    age = st.number_input("Age (years)", min_value=1, value=25, step=1)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    activity_level = st.selectbox(
        "Activity Level",
        options=[
            "Sedentary",
            "Lightly Active",
            "Moderately Active",
            "Very Active",
        ],
    )

    user_bmr = calculate_bmr(weight, height, age, gender)
    user_tdee = calculate_tdee(float(user_bmr), activity_level)

    if st.button("Calculate"):
        st.write("### Your Health Metrics")
        st.write(f"**BMI:** {weight / (height ** 2):.2f}")
        st.write(f"**BMR:** {user_bmr:.2f} kcal")
        st.write(f"**TDEE:** {user_tdee:.2f} kcal")

    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"]
    )

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
        except Exception:
            st.error("Could not open that image. Please upload a valid JPG or PNG.")
            return

        st.image(image, caption="Uploaded Image", use_container_width=True)

        temp_path = None
        try:
            suffix = ".png" if uploaded_image.name.lower().endswith(".png") else ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                image.save(temp_file, format="PNG" if suffix == ".png" else "JPEG")
                temp_path = temp_file.name

            if st.button("Detect"):
                with st.spinner("Detecting food..."):
                    detected_food = detect_image(temp_path)
