# Anomaly Detection using Reconstruction Loss from VAE

This project explores anomaly detection (AD) on the MNIST dataset using Variational Autoencoders (VAEs). The aim is to classify anomalies based on reconstruction loss, comparing two model architectures: **Linear** and **Convolutional**.

## Methodology

- **Dataset**: The MNIST dataset is used, where images of all digits except one are treated as the normal set, and the excluded digit is considered an anomaly.  
- **Approach**:  
  - Train a Variational Autoencoder (VAE) on the normal set to reconstruct images.  
  - Use reconstruction loss (difference between the input and reconstructed image) to detect anomalies.  
  - Anomalies are identified based on marginal p-values calculated using a validation set of normal images.  
- **Architectures**: Two types of models are implemented and compared:  
  - **Linear VAE**: Fully connected layers.  
  - **Convolutional VAE**: Convolutional layers for better feature extraction.  

## How to Run the Experiments

1. **Train and Test Models**:  
   - Linear Model:  
     ```bash
     python VAEAD.py
     ```
   - Convolutional Model:  
     ```bash
     python CVAEAD.py
     ```

2. **Visualize Results**:  
   Use Streamlit to visualize the results interactively:  
   ```bash
   streamlit run visu.py
   ```