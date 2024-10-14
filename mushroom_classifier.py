import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Load the model
model = joblib.load("stacking_model.pkl")

# Define the full names for the Cap Shape options
cap_shape_full_names = {
    "b": "Bell",
    "c": "Conical",
    "f": "Flat",
    "k": "Knobbed",
    "s": "Sunken",
    "x": "Convex",
}

# Define the full names for the Cap Surface options
cap_surface_full_names = {
    "f": "Fibrous",
    "g": "Grooves",
    "s": "Smooth",
    "y": "Scaly",
}

# Define the full names for the Cap Color options
cap_color_full_names = {
    "n": "Brown",
    "b": "Buff",
    "c": "Cinnamon",
    "g": "Gray",
    "r": "Green",
    "p": "Pink",
    "u": "Purple",
    "e": "Red",
    "w": "White",
    "y": "Yellow",
}

# Define the full names for other options similarly...

# Define the full names for the Bruises options
bruises_full_names = {
    "t": "Bruises",
    "f": "No Bruises",
}

# Define the full names for the Odor options
odor_full_names = {
    "a": "Almond",
    "l": "Anise",
    "c": "Creosote",
    "y": "Fishy",
    "f": "Foul",
    "m": "Musty",
    "n": "None",
    "p": "Pungent",
    "s": "Spicy",
}

# Define the full names for the Gill Attachment options
gill_attachment_full_names = {
    "a": "Attached",
    "d": "Descending",
    "f": "Free",
    "n": "Notched",
}

# Define the full names for the Gill Spacing options
gill_spacing_full_names = {
    "c": "Close",
    "w": "Crowded",
    "d": "Distant",
}

# Define the full names for the Gill Size options
gill_size_full_names = {
    "b": "Broad",
    "n": "Narrow",
}

# Define the full names for the Gill Color options
gill_color_full_names = {
    "k": "Black",
    "n": "Brown",
    "b": "Buff",
    "h": "Chocolate",
    "g": "Gray",
    "r": "Green",
    "o": "Orange",
    "p": "Pink",
    "u": "Purple",
    "e": "Red",
    "w": "White",
    "y": "Yellow",
}

# Define the full names for the Stalk Shape options
stalk_shape_full_names = {
    "e": "Enlarging",
    "t": "Tapering",
}

# Define the full names for the Stalk Root options
stalk_root_full_names = {
    "b": "Bulbous",
    "c": "Club",
    "u": "Cup",
    "e": "Equal",
    "z": "Rhizomorphs",
    "r": "Rooted",
    "?": "Missing",
}

# Define the full names for the Stalk Surface Above Ring options
stalk_surface_above_ring_full_names = {
    "f": "Fibrous",
    "y": "Scaly",
    "k": "Silky",
    "s": "Smooth",
}

# Define the full names for the Stalk Surface Below Ring options
stalk_surface_below_ring_full_names = {
    "f": "Fibrous",
    "y": "Scaly",
    "k": "Silky",
    "s": "Smooth",
}

# Define the full names for the Stalk Color Above Ring options
stalk_color_above_ring_full_names = {
    "n": "Brown",
    "b": "Buff",
    "c": "Cinnamon",
    "g": "Gray",
    "o": "Orange",
    "p": "Pink",
    "e": "Red",
    "w": "White",
    "y": "Yellow",
}

# Define the full names for the Stalk Color Below Ring options
stalk_color_below_ring_full_names = {
    "n": "Brown",
    "b": "Buff",
    "c": "Cinnamon",
    "g": "Gray",
    "o": "Orange",
    "p": "Pink",
    "e": "Red",
    "w": "White",
    "y": "Yellow",
}

# Define the full names for the Veil Type options
veil_type_full_names = {
    "p": "Partial",
    "u": "Universal",
}

# Define the full names for the Veil Color options
veil_color_full_names = {
    "n": "Brown",
    "o": "Orange",
    "w": "White",
    "y": "Yellow",
}

# Define the full names for the Ring Number options
ring_number_full_names = {
    "n": "None",
    "o": "One",
    "t": "Two",
}

# Define the full names for the Ring Type options
ring_type_full_names = {
    "c": "Cobwebby",
    "e": "Evanescent",
    "f": "Flaring",
    "l": "Large",
    "n": "None",
    "p": "Pendant",
    "s": "Sheathing",
    "z": "Zone",
}

# Define the full names for the Spore Print Color options
spore_print_color_full_names = {
    "k": "Black",
    "n": "Brown",
    "b": "Buff",
    "h": "Chocolate",
    "r": "Green",
    "o": "Orange",
    "u": "Purple",
    "w": "White",
    "y": "Yellow",
}

# Define the full names for the Population options
population_full_names = {
    "a": "Abundant",
    "c": "Clustered",
    "n": "Numerous",
    "s": "Scattered",
    "v": "Several",
    "y": "Solitary",
}

# Define the full names for the Habitat options
habitat_full_names = {
    "g": "Grasses",
    "l": "Leaves",
    "m": "Meadows",
    "p": "Paths",
    "u": "Urban",
    "w": "Waste",
    "d": "Woods",
}


# Define the PCA transformation function
def apply_pca(df):
    # Columns for each group
    cap_columns = ['cap-shape', 'cap-surface', 'cap-color']
    gill_columns = ['gill-attachment', 'gill-spacing', 'gill-size', 'gill-color']
    stalk_columns = ['stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring']
    scaler = StandardScaler()

    df[cap_columns] = scaler.fit_transform(df[cap_columns])
    df[gill_columns] = scaler.fit_transform(df[gill_columns])
    df[stalk_columns] = scaler.fit_transform(df[stalk_columns])
    pca = PCA(n_components=1)

    df['cap'] = pca.fit_transform(df[cap_columns])
    df['gill'] = pca.fit_transform(df[gill_columns])
    df['stalk'] = pca.fit_transform(df[stalk_columns])
    df = df.drop(columns=cap_columns + gill_columns + stalk_columns)

    return df

# Define the classification function
def classify_mushroom(data):
    transformed_data = apply_pca(data)
    prediction = model.predict(transformed_data)
    # Convert numeric predictions to class labels
    class_labels = ["edible" if pred == 0 else "poisonous" for pred in prediction]
    return class_labels


# Define the Streamlit app
def main():

    st.title("Mushroom Classifier")
    st.write("This app classifies mushrooms into edible or poisonous.")

    # Collect user input
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

    ring_number = st.selectbox("Ring Number", list(ring_number_full_names.values()))

    ring_type_selected = st.selectbox("Ring Type", list(ring_type_full_names.values()))
    ring_type = list(ring_type_full_names.keys())[list(ring_type_full_names.values()).index(ring_type_selected)]

    spore_print_color_selected = st.selectbox("Spore Print Color", list(spore_print_color_full_names.values()))
    spore_print_color = list(spore_print_color_full_names.keys())[list(spore_print_color_full_names.values()).index(spore_print_color_selected)]

    population_selected = st.selectbox("Population", list(population_full_names.values()))
    population = list(population_full_names.keys())[list(population_full_names.values()).index(population_selected)]

    habitat_selected = st.selectbox("Habitat", list(habitat_full_names.values()))
    habitat = list(habitat_full_names.keys())[list(habitat_full_names.values()).index(habitat_selected)]

    # Create a DataFrame from user input
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


    # Label encode the DataFrame
    input_data_encoded = input_data.apply(lambda x: pd.factorize(x)[0])

    # Check if the "Classify" button is clicked
    if st.button("Classify"):
        # Perform classification
        prediction = classify_mushroom(input_data_encoded)

        # Display result
        st.subheader("Prediction:")
        if prediction[0] == 1:
            st.header("The mushroom is **Edible**.")
        else:
            st.header("The mushroom is **Poisonous**.")

if __name__ == "__main__":
    main()
