import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json

st.set_page_config(layout="wide")

st.title("Anomaly Detections Using Reconstruction Loss from VAE and Marginal P-Values")
st.divider()

st.header("Model Info:")

col1, col2 = st.columns(2)
with col1:

    st.subheader("Linear VAE :")
    st.subheader("784px -> 512 -> 256 -> 2 -> 256 -> 512 -> 784")

with col2:
    st.subheader("Convolutional VAE :")
    st.subheader("1 -> 64 -> 128 -> 10 -> 128 -> 64 -> 1")

st.divider()

st.sidebar.title('Parameters: ')
anormal = st.sidebar.slider("Anomaly digit: ", min_value=0, max_value=9, value=0, step=1)
threshold = st.sidebar.slider("Threshold: ", min_value=0., max_value=0.2, step=0.01, value=0.05)

before_training = Image.open(f"figures/Anomaly_{anormal}/gen2_before_training.jpg")
after_training = Image.open(f"figures/Anomaly_{anormal}/gen2_after_training.jpg")

before_training_conv = Image.open(f"figures/Anomaly_{anormal}/conv2_gen_before_training.jpg")
after_training_conv = Image.open(f"figures/Anomaly_{anormal}/conv2_gen_after_training.jpg")

st.header("Digits generation before the training : ")
col1, col2 = st.columns(2)
with col1:
    st.image(before_training)
with col2:
    st.image(before_training_conv)
st.divider()
st.header("Digits generation after the training")
col1, col2 = st.columns(2)
with col1:
    st.image(after_training)
with col2:
    st.image(after_training_conv)

st.divider()

st.header('Mean scores for each digits :')
mean_scores = Image.open(f"figures/Anomaly_{anormal}/mean2_scores.jpg")
mean_scores_conv = Image.open(f"figures/Anomaly_{anormal}/conv2_mean_scores.jpg")
col1, col2 = st.columns(2)
with col1:
    st.image(mean_scores)
with col2:
    st.image(mean_scores_conv)
st.divider()

st.header('Statistical tests')

with open(f"p_values/2_{anormal}.json", "r") as file:
    p_values_test = json.load(file)

with open(f"p_values/conv2_{anormal}.json", "r") as file:
    p_values_test_conv = json.load(file)


results = []
for digit in range(10):
    test_p_values, len_test = p_values_test[str(digit)]
    test_p_values = np.asarray(test_p_values)

    test_p_values_conv, len_test_conv = p_values_test_conv[str(digit)]
    test_p_values_conv = np.asarray(test_p_values_conv)

    n_rejets = (test_p_values < threshold).sum().item()
    percentage_rejected = n_rejets / len_test

    n_rejets_conv = (test_p_values_conv < threshold).sum().item()
    percentage_rejected_conv = n_rejets_conv / len_test_conv

    # Ajouter les données au tableau
    results.append({
        "Digit": digit,
        "Anormal": "Yes" if digit == anormal else "No",
        "Threshold": threshold,
        "Rejections (linear)": f"{n_rejets}/{len_test}",
        "Rejection Rate (linear)": f"{percentage_rejected:.3%}",
        "Rejections (conv)": f"{n_rejets_conv}/{len_test_conv}",
        "Rejection Rate (conv)": f"{percentage_rejected_conv:.3%}"
    })

# Convertir les résultats en DataFrame pour l'affichage
df_results = pd.DataFrame(results)

# Afficher le tableau dans Streamlit
st.table(df_results)
