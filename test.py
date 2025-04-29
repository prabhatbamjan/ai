import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="PCA Image Compression", layout="wide")
st.title("📷 PCA Image Compression and Channel Analysis")

uploaded_file = st.file_uploader("Upload an image (e.g., Lenna)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    image_rgb = image.convert("RGB")
    image_array = np.array(image_rgb)

    # Show Red, Green, Blue Channels
    st.subheader("Color Channels (RGB)")
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(red_channel, caption="Red Channel", clamp=True)
    with col2:
        st.image(green_channel, caption="Green Channel", clamp=True)
    with col3:
        st.image(blue_channel, caption="Blue Channel", clamp=True)

    # Grayscale conversion
    image_gray = image.convert("L")
    image_array_gray = np.array(image_gray)
    st.subheader("Grayscale Image")
    st.image(image_array_gray, caption="Grayscale", clamp=True, channels="GRAY")

    # Middle section crop
    width, height = image_gray.size
    start_y = (height - 150) // 2
    cropped_image = image_gray.crop((0, start_y, width, start_y + 150))
    st.subheader("Middle Cropped Section")
    st.image(cropped_image, caption="Middle 150-pixel Crop", use_column_width=True)

    # Threshold
    threshold_image = np.where(image_array_gray < 100, 0, 255).astype(np.uint8)
    st.subheader("Binary Threshold (Threshold = 100)")
    st.image(threshold_image, caption="Binary Image", clamp=True)

    # Rotate
    rotated_image = image_gray.rotate(-90)
    st.subheader("Rotated Image (90° Clockwise)")
    st.image(rotated_image, caption="Rotated", use_column_width=True)

    # RGB from Grayscale
    rgb_image = np.stack((image_array_gray,) * 3, axis=-1)
    rgb_image = Image.fromarray(rgb_image)
    st.subheader("Grayscale Converted to RGB")
    st.image(rgb_image, caption="RGB from Grayscale", use_column_width=True)

    # PCA Section
    st.subheader("📉 PCA Compression")

    # Standardize
    mean = np.mean(image_array_gray, axis=0)
    std = np.std(image_array_gray, axis=0)
    standardized_data = (image_array_gray - mean) / std

    # Covariance + Eigen
    cov_matrix = np.cov(standardized_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Plot explained variance
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(eigenvalues) / np.sum(eigenvalues))
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Explained Variance vs. # Components")
    st.pyplot(fig)

    # Select k values
    k = st.slider("Select number of PCA components (k)", 5, min(image_array_gray.shape), 50, step=5)

    def reconstruct_image(k_val, eigvecs, std_data, mean, std):
        top_k_eigvecs = eigvecs[:, :k_val]
        reduced = np.dot(std_data, top_k_eigvecs)
        reconstructed = np.dot(reduced, top_k_eigvecs.T)
        return (reconstructed * std) + mean

    recon_image = reconstruct_image(k, eigenvectors, standardized_data, mean, std)
    st.subheader(f"Reconstructed Image using {k} Principal Components")
    st.image(recon_image, clamp=True, caption=f"k = {k}", use_column_width=True)
