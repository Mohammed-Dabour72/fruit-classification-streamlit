import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Define the labels
LABELS = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',
 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion',
 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato',
 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
 'turnip', 'watermelon']


# Load the saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.h5")

model = load_model()

# Function to preprocess the uploaded image
def preprocess_image(image):
    """
    Preprocess the image to match the model's input requirements.
    - Resize the image to the model's input size (224x224).
    - Normalize pixel values to range [0, 1].
    - Add a batch dimension.
    """
    img = image.resize((224, 224))  # Resize to model input size
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Streamlit App Interface
st.title("Image Classification App")
st.write("Upload an image to classify it into one of the following categories:")
st.write(", ".join(LABELS))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    with st.spinner("Classifying..."):
        try:
            processed_image = preprocess_image(image)
            
            # Debugging: Show processed image shape
            st.write("Processed Image Shape:", processed_image.shape)
            
            # Model prediction
            prediction = model.predict(processed_image)
            
            # Debugging: Log raw predictions
            st.write("Raw Predictions:", prediction)
            
            # Get the predicted label
            predicted_label = LABELS[np.argmax(prediction)]

            # Display the prediction
            st.success(f"Prediction: {predicted_label}")
            
            # Display confidence scores as a JSON object
            confidence_scores = {label: float(score) for label, score in zip(LABELS, prediction[0])}
            st.write("Confidence Scores:")
            st.json(confidence_scores)
        except Exception as e:
            st.error(f"Error during classification: {e}")
