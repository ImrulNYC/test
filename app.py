import os
import torch
import boto3
import streamlit as st
from transformers import AutoConfig, AutoModelForImageClassification, ViTFeatureExtractor
from safetensors.torch import load_file
from PIL import Image

# AWS S3 Configuration using Streamlit secrets
aws_access_key_id = st.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws_secret_access_key"]
region_name = st.secrets["region_name"]

# S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# Bucket and model keys
BUCKET_NAME = "flowerm"
MODEL_KEY = "model.safetensors"
CONFIG_KEY = "config.json"
PREPROCESSOR_KEY = "preprocessor_config.json"

# Temporary paths for downloaded files
model_path = "/tmp/model.safetensors"
config_path = "/tmp/config.json"
preprocessor_path = "/tmp/preprocessor_config.json"

# Function to download model components from S3
def download_file_from_s3(bucket_name, s3_key, local_path):
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        print(f"Downloaded {s3_key} successfully.")
    except Exception as e:
        st.error(f"Failed to download {s3_key} from S3: {str(e)}")
        raise

# Streamlit app setup
st.title("Flower Identification App 🌼")
st.write("Upload an image of a flower to identify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Download the model components from S3
    with st.spinner("Downloading model files..."):
        download_file_from_s3(BUCKET_NAME, MODEL_KEY, model_path)
        download_file_from_s3(BUCKET_NAME, CONFIG_KEY, config_path)
        download_file_from_s3(BUCKET_NAME, PREPROCESSOR_KEY, preprocessor_path)

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
    def predict_flower(img_path):
        image = Image.open(img_path).convert("RGB")
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
        predicted_label, confidence = predict_flower(uploaded_file)

    # Display prediction results
    if predicted_label:
        st.success(f"Predicted Flower: {predicted_label} with {confidence:.2f}% confidence.")
    else:
        st.warning("The flower cannot be confidently recognized. Please try another image.")

