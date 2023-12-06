#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import joblib
import pandas as pd

# Load your model and preprocessor
model = joblib.load('music_recommendation_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Streamlit app title
st.title('Music Genre Recommendation App')

# Creating input fields for user data
age = st.number_input('Age', min_value=1, max_value=100, value=30)
primary_streaming_service = st.selectbox('Primary Streaming Service', ['Spotify', 'Apple Music', 'Other'], index=0)
hours_per_day = st.number_input('Hours per Day', min_value=0, max_value=24, value=3)
while_working = st.radio('Listen While Working', ('Yes', 'No'), index=1)
instrumentalist = st.radio('Instrumentalist', ('Yes', 'No'), index=1)
composer = st.radio('Composer', ('Yes', 'No'), index=1)
exploratory = st.radio('Exploratory', ('Yes', 'No'), index=0)
foreign_languages = st.radio('Foreign Languages', ('Yes', 'No'), index=0)
bpm = st.slider('Beats Per Minute (BPM)', min_value=0, max_value=300, value=120)
anxiety = st.slider('Anxiety Level', min_value=0, max_value=10, value=5)
depression = st.slider('Depression Level', min_value=0, max_value=10, value=3)
insomnia = st.slider('Insomnia Level', min_value=0, max_value=10, value=2)
ocd = st.slider('OCD Level', min_value=0, max_value=10, value=1)

# Button to make prediction
if st.button('Recommend Genre'):
    user_data = {
        'Age': age,
        'Primary streaming service': primary_streaming_service,
        'Hours per day': hours_per_day,
        'While working': while_working,
        'Instrumentalist': instrumentalist,
        'Composer': composer,
        'Exploratory': exploratory,
        'Foreign languages': foreign_languages,
        'BPM': bpm,
        'Anxiety': anxiety,
        'Depression': depression,
        'Insomnia': insomnia,
        'OCD': ocd
    }

    # Convert user data to DataFrame
    user_df = pd.DataFrame([user_data])

    # Preprocess the data
    transformed_data = preprocessor.transform(user_df)

    # Make a prediction
    recommendation = model.predict(transformed_data)

    # Display the recommendation
    st.success(f"Recommended Music Genre: {recommendation[0]}")


# In[ ]:




