import os
import torch
import streamlit as st
from transformers import AutoConfig, AutoModelForImageClassification, ViTFeatureExtractor
from safetensors.torch import load_file
from PIL import Image
import urllib.request
import io

# Bucket and model keys
BUCKET_URL = "https://flowerm.s3.amazonaws.com/"
MODEL_KEY = "model.safetensors"
CONFIG_KEY = "config.json"
PREPROCESSOR_KEY = "preprocessor_config.json"

# Temporary paths for downloaded files
model_path = "/tmp/model.safetensors"
config_path = "/tmp/config.json"
preprocessor_path = "/tmp/preprocessor_config.json"

# Function to download files from a public S3 URL
def download_file_from_s3(url, local_path):
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded {url} successfully.")
    except Exception as e:
        st.error(f"Failed to download {url}: {str(e)}")
        raise

# Streamlit app setup
st.title("Flower Identification App 🌼")
st.write("Upload an image of a flower to identify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Convert uploaded file to a format compatible with PIL
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Download the model components from S3
        with st.spinner("Downloading model files..."):
            download_file_from_s3(f"{BUCKET_URL}{MODEL_KEY}", model_path)
            download_file_from_s3(f"{BUCKET_URL}{CONFIG_KEY}", config_path)
            download_file_from_s3(f"{BUCKET_URL}{PREPROCESSOR_KEY}", preprocessor_path)

        # Load model configuration and preprocessor
        config = AutoConfig.from_pretrained(config_path)
        preprocessor = ViTFeatureExtractor.from_pretrained(preprocessor_path)

        # Load model weights using safetensors
        state_dict = load_file(model_path)
        model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=state_dict
        )

        # Label mappings
        id_to_label = {
            0: 'calendula', 1: 'coreopsis', 2: 'rose', 3: 'black_eyed_susan', 4: 'water_lily', 5: 'california_poppy',
            6: 'dandelion', 7: 'magnolia', 8: 'astilbe', 9: 'sunflower', 10: 'tulip', 11: 'bellflower',
            12: 'iris', 13: 'common_daisy', 14: 'daffodil', 15: 'carnation'
        }

        # Predict the flower type
        def predict_flower(image):
            inputs = preprocessor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence = torch.max(probabilities).item() * 100
                predicted_class = torch.argmax(probabilities, dim=1).item()

            predicted_label = id_to_label.get(predicted_class, "Unknown")

            if confidence >= 80:
                return predicted_label, confidence
            else:
                return None, None

        # Make prediction
        with st.spinner("Predicting flower type..."):
            predicted_label, confidence = predict_flower(image)

        # Display prediction results
        if predicted_label:
            st.success(f"Predicted Flower: {predicted_label} with {confidence:.2f}% confidence.")
        else:
            st.warning("The flower cannot be confidently recognized. Please try another image.")

    except Image.UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a valid JPG, JPEG, or PNG file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

