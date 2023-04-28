import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('C:/Users/bibhu/OneDrive/Desktop/FinalProject/saved_models/cnn-softmax-adam.h5')

class_names = ['no', 'yes']

def predict_tumor_class(model, image):
    # Normalize the image
    image = image.astype('float32') / 255.0

    # Add a batch dimension
    image = np.expand_dims(image, axis=0)

    # Make the prediction
    predictions = model.predict(image)

    # Get the predicted label and confidence
    predicted_label = class_names[np.argmax(predictions[0])]
    confidence = round(np.max(predictions[0]) * 100, 2)

    return predicted_label, confidence

def main():
    st.set_page_config(page_title="Brain Tumor Detection", page_icon=":brain:")

    st.title("Brain Tumor Detection")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image).resize((256,256))
        st.image(image, caption='Uploaded Image',  width=200)

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Predict the tumor class
        predicted_label, confidence = predict_tumor_class(model, image_array)

        # Display the predicted label and confidence
        st.success(f"Tumor Prediction: {predicted_label}.   Confidence: {confidence}%")

if __name__ == '__main__':
    main()
