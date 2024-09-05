import streamlit as st
import requests
import json

import os
from dotenv import load_dotenv

load_dotenv()

# Set API endpoint and API key
API_ENDPOINT = "http://api.openweathermap.org/data/2.5/weather?"
API_KEY =  os.getenv("APIKEY")

# Streamlit page 

# Configuring Streamlit GUI 
st.set_page_config(page_title='Weather')
st.title(':violet[Weather Dashboard]')

st.header(' ')
st.subheader("Enter a city")

# Get user input 
city = st.text_input("City")

# Make API call to get weather data
if city:
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    response = requests.get(API_ENDPOINT, params=params)
    if response.status_code == 200:
        # Parse JSON response
        data = json.loads(response.content)

        # Extract relevant data
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

        # Display data in a user-friendly format
        st.write(f"Current weather in {city}:")
        st.write(f"Temperature : {temp}°C")
        st.write(f"Temperature feels like : {feels_like}°C")
        st.write(f"Humidity : {humidity}%")
        st.write(f"Description : {description}")
    else:
        st.error("Error fetching weather data. Please try again.")
else:
    st.write("Please enter a city to get the weather")