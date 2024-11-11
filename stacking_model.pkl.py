import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


# Load the trained model and scaler (make sure the paths are correct)
model = joblib.load('C:/streamlit/stacking_model.pkl')
scaler = joblib.load('C:/streamlit/scaler.pkl')


# Example PCA transformation function (should match your training setup)
def apply_pca(df, pca_model):
    """
    Apply scaling and PCA transformation.
    Ensure this matches how you trained the model.
    """
    # Scale the input data using the saved scaler
    df_scaled = scaler.transform(df)
    
    # Apply PCA transformation using the pre-trained PCA model
    df_pca = pca_model.transform(df_scaled)
    return df_pca


# Define the classification function
def classify_mushroom(data, model, pca_model):
    """
    Classify the mushroom using the trained model.
    """
    transformed_data = apply_pca(data, pca_model)  # Apply PCA and scaling
    prediction = model.predict(transformed_data)  # Predict with the trained model
    # Convert numeric predictions to class labels
    class_labels = ["Edible" if pred == 0 else "Poisonous" for pred in prediction]
    return class_labels


# Define the full names for the mushroom features
# (your full lists for the mushroom features)
cap_shape_full_names = {
    "b": "Bell", "c": "Conical", "f": "Flat", "k": "Knobbed", "s": "Sunken", "x": "Convex"
}
# Add similar definitions for other categories like cap_surface_full_names, cap_color_full_names etc.

# Loading PCA model (ensure this path points to your actual PCA model if saved separately)
pca_model = joblib.load('C:/streamlit/pca_model.pkl')


# Streamlit interface for user input
def main():
    st.title("Mushroom Classifier")
    st.write("This app classifies mushrooms into edible or poisonous.")

    # Collect user input for each mushroom characteristic
    cap_shape_selected = st.selectbox("Cap Shape", list(cap_shape_full_names.values()))
    cap_shape = list(cap_shape_full_names.keys())[list(cap_shape_full_names.values()).index(cap_shape_selected)]

    cap_surface_selected = st.selectbox("Cap Surface", list(cap_surface_full_names.values()))
    cap_surface = list(cap_surface_full_names.keys())[list(cap_surface_full_names.values()).index(cap_surface_selected)]

    cap_color_selected = st.selectbox("Cap Color", list(cap_color_full_names.values()))
    cap_color = list(cap_color_full_names.keys())[list(cap_color_full_names.values()).index(cap_color_selected)]

    bruises = st.selectbox("Bruises", list(bruises_full_names.values()))

    odor_selected = st.selectbox("Odor", list(odor_full_names.values()))
    odor = list(odor_full_names.keys())[list(odor_full_names.values()).index(odor_selected)]

    # Similarly, collect all other feature inputs

    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'cap-shape': [cap_shape],
        'cap-surface': [cap_surface],
        'cap-color': [cap_color],
        'bruises': [bruises],
        'odor': [odor],
        # Add other feature columns here (similar to the above pattern)
    })

    # Label encode the categorical input data
    input_data_encoded = input_data.apply(lambda x: pd.factorize(x)[0])

    # Ensure PCA and scaling are applied using pre-trained models
    if st.button("Classify"):
        prediction = classify_mushroom(input_data_encoded, model, pca_model)
        
        # Display result
        st.subheader("Prediction:")
        if prediction[0] == "Edible":
            st.header("The mushroom is **Edible**.")
        else:
            st.header("The mushroom is **Poisonous**.")

if __name__ == "__main__":
    main()
