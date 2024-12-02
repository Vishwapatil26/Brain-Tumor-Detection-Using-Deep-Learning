import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("Brain_Tumor10EpochsCategorical.h5")  # Replace with your model path

# Function to process the image and make prediction
def predict_image(img):
    # Resize image to the model's input size (e.g., 64x64)
    img = img.resize((64, 64))
    
    # Convert image to array and normalize
    img_array = image.img_to_array(img) / 255.0
    
    # Add batch dimension (expand dims to shape (1, 64, 64, 3))
    input_img = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(input_img)
    
    # Return predicted class (0 for No Cancer, 1 for Cancer)
    return predictions

# Streamlit UI
st.title("Brain Tumor Detection")
st.write("Upload an image of a brain scan to predict whether it has cancer or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Display the image size and type for debugging
    st.write(f"Image size: {img.size}, Image type: {type(img)}")

    # Predict the result when the user presses the button
    if st.button("Predict"):
        predictions = predict_image(img)
        
        # Display the raw predictions for debugging
        st.write(f"Raw predictions: {predictions}")
        
        # Output prediction result
        if np.argmax(predictions) == 1:
            st.write("Prediction: Cancer Detected!")
        else:
            st.write("Prediction: No Cancer Detected!")
