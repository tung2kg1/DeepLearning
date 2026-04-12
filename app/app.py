import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# =====================
# CLASS UTIL
# =====================
def prettify(name):
    return name.replace("___", " - ").replace("_", " ").title()

# =====================
# DEFAULT DATASET CLASSES
# =====================
plantvillage_default_classes = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
    'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

cifar10_default_classes = [
    "Airplane","Automobile","Bird","Cat","Deer",
    "Dog","Frog","Horse","Ship","Truck"
]

catdog_default_classes = ["Cat", "Dog"]

# =====================
# LOAD MODELS
# =====================
@st.cache_resource
def load_models():
    plant_path = "plant_village.h5"
    catdog_path = "catvsdog_model.h5"
    cifar_path = "cifar10_model.h5"

    plant_model = tf.keras.models.load_model(plant_path)
    catdog_model = tf.keras.models.load_model(catdog_path)
    cifar_model = tf.keras.models.load_model(cifar_path)

    # 🔥 Không dùng JSON nữa — dùng mapping chuẩn theo dataset
    plant_classes = [prettify(c) for c in plantvillage_default_classes]
    catdog_classes = catdog_default_classes
    cifar_classes = cifar10_default_classes

    return (plant_model, plant_classes,
            catdog_model, catdog_classes,
            cifar_model, cifar_classes)

(plant_model, plant_classes,
 catdog_model, catdog_classes,
 cifar_model, cifar_classes) = load_models()

# =====================
# PREPROCESS
# =====================
def preprocess(image, model):
    # 🔥 Auto detect input size từ model (fix triệt để bug 224 vs 128)
    try:
        input_shape = model.input_shape
        if isinstance(input_shape, list):  # model multi-input
            input_shape = input_shape[0]
        size = input_shape[1]
    except Exception:
        size = 224  # fallback an toàn

    img = image.resize((size, size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =====================
# PREDICT
# =====================
def predict(model, image, classes, model_name=None):
    # 🔥 đảm bảo luôn resize đúng theo model
    img = preprocess(image, model)
    preds = model.predict(img)

    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    num_outputs = preds.shape[1]

    # 🔥 Fix mismatch classes vs output
    if not classes or len(classes) != num_outputs:
        if model_name == "Plant Disease":
            classes = [prettify(c) for c in plantvillage_default_classes]
        elif model_name == "Cat vs Dog":
            classes = catdog_default_classes
        elif model_name == "CIFAR-10":
            classes = cifar10_default_classes

    # Nếu hợp lệ
    if classes and class_idx < len(classes):
        return classes[class_idx], confidence

    # fallback cuối cùng (hiếm)
    return f"Class {class_idx}", confidence

# =====================
# UI
# =====================
st.title("🧠 Multi Model Image Classifier (Auto Class Detection)")
st.write("Upload ảnh và chọn model để dự đoán")

model_option = st.selectbox(
    "Chọn model",
    ["Plant Disease", "Cat vs Dog", "CIFAR-10"]
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    if st.button("Predict"):
        if model_option == "Plant Disease":
            label, conf = predict(plant_model, image, plant_classes)
        elif model_option == "Cat vs Dog":
            label, conf = predict(catdog_model, image, catdog_classes)
        else:
            label, conf = predict(cifar_model, image, cifar_classes)

        st.success(f"Prediction: {label} ({conf*100:.2f}%)")

# =====================
# FOOTER
# =====================
st.markdown("---")
st.write("Built with Streamlit + TensorFlow 🚀")
