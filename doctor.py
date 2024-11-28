import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import folium
from streamlit_folium import st_folium
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import PorterStemmer

# Initialize PorterStemmer
ps = PorterStemmer()

# Load pre-trained models and data
doc_df = pickle.load(open('doc_df.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
vector = pickle.load(open('vector.pkl', 'rb'))
description_df = pickle.load(open('description_df.pkl', 'rb'))
specialist_df = pickle.load(open('specialist_df.pkl', 'rb'))

# Initialize session state
if 'selected_doctor' not in st.session_state:
    st.session_state.selected_doctor = None

if 'doctors' not in st.session_state:
    st.session_state.doctors = []

# Text processing function
def transform_data(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Filter input text
def filter_input(text):
    text = transform_data(text)

    # Repeat words to increase weight of user input
    words = text.split()
    tripled_words = ' '.join(words * 3)
    input_vector = tfidf.transform([tripled_words]).toarray()
    return input_vector

# Function to recommend doctors
def recommend_doctor_with_map(inp_vector, vector, user_pincode):
    # Step 1: Symptom-based filtering
    similarity_scores = cosine_similarity(inp_vector, vector)
    sorted_indices = similarity_scores.argsort()[0][::-1]

    similar_disease = sorted_indices[:2]
    top_disease = description_df.iloc[similar_disease]

    # Create list of predicted diseases
    predicted_disease = []
    for disease in top_disease.itertuples():
        predicted_disease.append(disease.Disease)

    # Step 2: Specialist list filtering
    specialist_list = []
    for idx, specialist in specialist_df['Drug Reaction'].items():
        if specialist in predicted_disease:
            specialist_list.append(specialist_df.loc[idx, 'Allergist'])

    # Step 3: Get the doctors who are specialists
    doctor_list = []
    for idx, doctors in doc_df['Specialist'].items():
        if doctors in specialist_list:
            doctor_list.append(doc_df.loc[idx, 'Name'])

    # Step 4: Location-based filtering
    doctor_distances = []
    for doctor_name in doctor_list:
        doctor_details = doc_df[doc_df['Name'] == doctor_name].iloc[0]
        distance = abs(int(doctor_details['Pincode']) - int(user_pincode))
        doctor_distances.append((doctor_details, distance))

    # Sort doctors by distance and return top 4
    doctor_distances.sort(key=lambda x: x[1])
    top_doctors = [doctor[0] for doctor in doctor_distances[:4]]

    return top_doctors

st.title("Doctor Recommender System")

# Input Section
user_symptoms = st.text_input("Enter your symptoms:")
user_pincode = st.text_input("Enter your pincode:")

if st.button("Find Doctors"):
    if user_symptoms and user_pincode:
        inp_vector = filter_input(user_symptoms)

        # Get doctor recommendations
        doctors = recommend_doctor_with_map(inp_vector, vector, user_pincode)
        st.session_state.doctors = doctors  # Store doctors in session state
    else:
        st.error("Please fill in both the symptoms and pincode!")

# Display doctor recommendations if available
if st.session_state.doctors:
    st.header("Recommended Doctors")
    selected_doctor = st.radio(
        "Select a doctor to view details and location:",
        [doctor['Name'] for doctor in st.session_state.doctors]
    )
    st.session_state.selected_doctor = selected_doctor  # Store selected doctor in session state

    if st.session_state.selected_doctor:
        doctor_details = next(
            doctor for doctor in st.session_state.doctors if doctor['Name'] == st.session_state.selected_doctor
        )
        st.write(f"Name: Dr. {doctor_details['Name']}")
        st.write(f"Specialist: {doctor_details['Specialist']}")
        st.write(f"Phone Number: {doctor_details['Phone Number']}")
        st.write(f"Timing: {doctor_details['Timing']}")
        st.write(f"Address: {doctor_details['Address']}")
        st.write(f"Pincode: {doctor_details['Pincode']}")

        # Generate map
        doctor_map = folium.Map(location=[doctor_details['Latitude'], doctor_details['Longitude']], zoom_start=14)
        folium.Marker(
            location=[doctor_details['Latitude'], doctor_details['Longitude']],
            popup=f"Doctor: {doctor_details['Name']}<br>Specialist: {doctor_details['Specialist']}<br>Phone: {doctor_details['Phone Number']}<br>Address: {doctor_details['Address']}<br>Pincode: {doctor_details['Pincode']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(doctor_map)

        # Display map
        st_folium(doctor_map, width=700)

        # Add a Navigate button to open Google Maps
        google_maps_url = f"https://www.google.com/maps/search/?api=1&query={doctor_details['Latitude']},{doctor_details['Longitude']}"

        st.markdown(f"[DIRECTION]({google_maps_url})", unsafe_allow_html=True)
