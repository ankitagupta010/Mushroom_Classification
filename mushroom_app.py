import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load the trained model
model = joblib.load("stacking_model.pkl")

# Define full names for mushroom features (ensure they match the ones used during training)
cap_shape_full_names = {"b": "Bell", "c": "Conical", "f": "Flat", "k": "Knobbed", "s": "Sunken", "x": "Convex"}
cap_surface_full_names = {"f": "Fibrous", "g": "Grooves", "s": "Smooth", "y": "Scaly"}
cap_color_full_names = {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "r": "Green", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"}
bruises_full_names = {"t": "Bruises", "f": "No Bruises"}
odor_full_names = {"a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy", "f": "Foul", "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy"}
gill_attachment_full_names = {"a": "Attached", "d": "Descending", "f": "Free", "n": "Notched"}
gill_spacing_full_names = {"c": "Close", "w": "Crowded", "d": "Distant"}
gill_size_full_names = {"b": "Broad", "n": "Narrow"}
gill_color_full_names = {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "g": "Gray", "r": "Green", "o": "Orange", "p": "Pink", "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"}
stalk_shape_full_names = {"e": "Enlarging", "t": "Tapering"}
stalk_root_full_names = {"b": "Bulbous", "c": "Club", "u": "Cup", "e": "Equal", "z": "Rhizomorphs", "r": "Rooted", "?": "Missing"}
stalk_surface_above_ring_full_names = {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"}
stalk_surface_below_ring_full_names = {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"}
stalk_color_above_ring_full_names = {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"}
stalk_color_below_ring_full_names = {"n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange", "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"}
ring_number_full_names = {"n": "None", "o": "One", "t": "Two"}
ring_type_full_names = {"c": "Cobwebby", "e": "Evanescent", "f": "Flaring", "l": "Large", "n": "None", "p": "Pendant", "s": "Sheathing", "z": "Zone"}
spore_print_color_full_names = {"k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate", "r": "Green", "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow"}
population_full_names = {"a": "Abundant", "c": "Clustered", "n": "Numerous", "s": "Scattered", "v": "Several", "y": "Solitary"}
habitat_full_names = {"g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths", "u": "Urban", "w": "Waste", "d": "Woods"}

# Function to encode categorical columns
def encode_categorical_columns(df):
    le = LabelEncoder()
    categorical_columns = [
        'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
        'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring',
        'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    
    return df

# Function to apply PCA transformations and scaling
def apply_pca(df):
    # Columns for each group
    cap_columns = ['cap-shape', 'cap-surface', 'cap-color']
    gill_columns = ['gill-attachment', 'gill-spacing', 'gill-size', 'gill-color']
    stalk_columns = ['stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring']
    
    scaler = StandardScaler()
    
    # Standardize the data
    df[cap_columns] = scaler.fit_transform(df[cap_columns])
    df[gill_columns] = scaler.fit_transform(df[gill_columns])
    df[stalk_columns] = scaler.fit_transform(df[stalk_columns])

    # Apply PCA
    pca = PCA(n_components=1)
    df['cap'] = pca.fit_transform(df[cap_columns])
    df['gill'] = pca.fit_transform(df[gill_columns])
    df['stalk'] = pca.fit_transform(df[stalk_columns])

    # Drop original columns after transformation
    df = df.drop(columns=cap_columns + gill_columns + stalk_columns)
    
    return df

# Function to classify mushroom
def classify_mushroom(data):
    encoded_data = encode_categorical_columns(data)
    transformed_data = apply_pca(encoded_data)
    prediction = model.predict(transformed_data)
    return ["edible" if pred == 0 else "poisonous" for pred in prediction]

# Define Streamlit app
def main():
    st.title("Mushroom Classifier")
    st.write("This app classifies mushrooms into edible or poisonous.")

    # User input for the mushroom features
    cap_shape_selected = st.selectbox("Cap Shape", list(cap_shape_full_names.values()))
    cap_shape = list(cap_shape_full_names.keys())[list(cap_shape_full_names.values()).index(cap_shape_selected)]

    cap_surface_selected = st.selectbox("Cap Surface", list(cap_surface_full_names.values()))
    cap_surface = list(cap_surface_full_names.keys())[list(cap_surface_full_names.values()).index(cap_surface_selected)]

    cap_color_selected = st.selectbox("Cap Color", list(cap_color_full_names.values()))
    cap_color = list(cap_color_full_names.keys())[list(cap_color_full_names.values()).index(cap_color_selected)]

    bruises = st.selectbox("Bruises", list(bruises_full_names.values()))

    odor_selected = st.selectbox("Odor", list(odor_full_names.values()))
    odor = list(odor_full_names.keys())[list(odor_full_names.values()).index(odor_selected)]

    gill_attachment_selected = st.selectbox("Gill Attachment", list(gill_attachment_full_names.values()))
    gill_attachment = list(gill_attachment_full_names.keys())[list(gill_attachment_full_names.values()).index(gill_attachment_selected)]

    gill_spacing_selected = st.selectbox("Gill Spacing", list(gill_spacing_full_names.values()))
    gill_spacing = list(gill_spacing_full_names.keys())[list(gill_spacing_full_names.values()).index(gill_spacing_selected)]

    gill_size = st.selectbox("Gill Size", list(gill_size_full_names.values()))

    gill_color_selected = st.selectbox("Gill Color", list(gill_color_full_names.values()))
    gill_color = list(gill_color_full_names.keys())[list(gill_color_full_names.values()).index(gill_color_selected)]

    stalk_shape_selected = st.selectbox("Stalk Shape", list(stalk_shape_full_names.values()))
    stalk_shape = list(stalk_shape_full_names.keys())[list(stalk_shape_full_names.values()).index(stalk_shape_selected)]

    stalk_root_selected = st.selectbox("Stalk Root", list(stalk_root_full_names.values()))
    stalk_root = list(stalk_root_full_names.keys())[list(stalk_root_full_names.values()).index(stalk_root_selected)]

    stalk_surface_above_ring_selected = st.selectbox("Stalk Surface Above Ring", list(stalk_surface_above_ring_full_names.values()))
    stalk_surface_above_ring = list(stalk_surface_above_ring_full_names.keys())[list(stalk_surface_above_ring_full_names.values()).index(stalk_surface_above_ring_selected)]

    stalk_surface_below_ring_selected = st.selectbox("Stalk Surface Below Ring", list(stalk_surface_below_ring_full_names.values()))
    stalk_surface_below_ring = list(stalk_surface_below_ring_full_names.keys())[list(stalk_surface_below_ring_full_names.values()).index(stalk_surface_below_ring_selected)]

    stalk_color_above_ring_selected = st.selectbox("Stalk Color Above Ring", list(stalk_color_above_ring_full_names.values()))
    stalk_color_above_ring = list(stalk_color_above_ring_full_names.keys())[list(stalk_color_above_ring_full_names.values()).index(stalk_color_above_ring_selected)]

    stalk_color_below_ring_selected = st.selectbox("Stalk Color Below Ring", list(stalk_color_below_ring_full_names.values()))
    stalk_color_below_ring = list(stalk_color_below_ring_full_names.keys())[list(stalk_color_below_ring_full_names.values()).index(stalk_color_below_ring_selected)]

    ring_number_selected = st.selectbox("Ring Number", list(ring_number_full_names.values()))
    ring_number = list(ring_number_full_names.keys())[list(ring_number_full_names.values()).index(ring_number_selected)]

    ring_type_selected = st.selectbox("Ring Type", list(ring_type_full_names.values()))
    ring_type = list(ring_type_full_names.keys())[list(ring_type_full_names.values()).index(ring_type_selected)]

    spore_print_color_selected = st.selectbox("Spore Print Color", list(spore_print_color_full_names.values()))
    spore_print_color = list(spore_print_color_full_names.keys())[list(spore_print_color_full_names.values()).index(spore_print_color_selected)]

    population_selected = st.selectbox("Population", list(population_full_names.values()))
    population = list(population_full_names.keys())[list(population_full_names.values()).index(population_selected)]

    habitat_selected = st.selectbox("Habitat", list(habitat_full_names.values()))
    habitat = list(habitat_full_names.keys())[list(habitat_full_names.values()).index(habitat_selected)]

    # Combine all inputs into a DataFrame
    input_data = pd.DataFrame({
        'cap-shape': [cap_shape],
        'cap-surface': [cap_surface],
        'cap-color': [cap_color],
        'bruises': [bruises],
        'odor': [odor],
        'gill-attachment': [gill_attachment],
        'gill-spacing': [gill_spacing],
        'gill-size': [gill_size],
        'gill-color': [gill_color],
        'stalk-shape': [stalk_shape],
        'stalk-root': [stalk_root],
        'stalk-surface-above-ring': [stalk_surface_above_ring],
        'stalk-surface-below-ring': [stalk_surface_below_ring],
        'stalk-color-above-ring': [stalk_color_above_ring],
        'stalk-color-below-ring': [stalk_color_below_ring],
        'ring-number': [ring_number],
        'ring-type': [ring_type],
        'spore-print-color': [spore_print_color],
        'population': [population],
        'habitat': [habitat]
    })

    if st.button("Classify"):
        result = classify_mushroom(input_data)
        st.write(f"The mushroom is: {result[0]}")
        
if __name__ == "__main__":
    main()
