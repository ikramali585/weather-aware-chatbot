import streamlit as st
import requests
import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

# --- Page Config ---
st.set_page_config(page_title="Crop Advisor", page_icon="🌾")

# --- Load environment variables ---

# Load .env file
load_dotenv()

# Retrieve API keys from environment variables
weather_api_key = os.getenv("WEATHER_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Set environment variable for Groq
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
    crop_bot = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    conversation = ConversationChain(llm=crop_bot)
else:
    st.warning("Please set Groq API key in the env.")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state and groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
    crop_bot = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    st.session_state.memory = ConversationBufferMemory()
    st.session_state.conversation = ConversationChain(
        llm=crop_bot, memory=st.session_state.memory
    )
# --- Main Inputs ---
st.title("🌦️ Crop Recommendation System")
crop = st.text_input("Enter your crop's name:")
city = st.text_input("Enter your city:")

# --- Filter Forecast Data ---
def filter_data(data):
    unique_dates = set()
    filtered_data = []
    for entry in data['list']:
        date = entry['dt_txt'][0:-9]
        if date not in unique_dates:
            unique_dates.add(date)
            filtered_data.append(entry)
    return filtered_data

# --- Check Forecast ---
def check_weather_forecast(city):
    ndays = 40
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&cnt={ndays}&appid={weather_api_key}"
    response = requests.get(url)
    if response.status_code != 200 or response.json().get('cod') == '404':
        return None, "City not found"

    data = response.json()
    filtered_data = filter_data(data)
    rain_threshold_mm = 1.6
    wind_speed_threshold_mph = 20
    high_temp_c = 35
    low_temp_c = 0

    worst_days = []

    for day in filtered_data:
        date = day['dt_txt']
        rain = day.get('rain', {}).get('3h', 0)
        wind_speed = day['wind']['speed']
        temperature = day['main']['temp'] - 273.15  # Kelvin to Celsius

        if rain >= rain_threshold_mm or wind_speed >= wind_speed_threshold_mph or temperature >= high_temp_c or temperature <= low_temp_c:
            worst_days.append(date)

    return worst_days, None

# --- Main Logic ---
if st.button("Get Initial Recommendation"):
    if not weather_api_key:
        st.error("Please enter Weather API key in the sidebar.")
    elif not crop or not city:
        st.warning("Please fill in both crop name and city.")
    else:
        weather_data = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=imperial&APPID={weather_api_key}"
        )
        data = weather_data.json()
        if data.get('cod') == '404':
            st.error("City not found!")
        else:
            weather = data['weather'][0]['main']
            temp = round(data['main']['temp'])
            st.success(f"Weather in {city}: {weather}, {temp}ºF")

            worst_days, error = check_weather_forecast(city)
            if error:
                st.error(error)
            else:
                st.write("🌧️ **Worst Weather Days:**")
                if worst_days:
                    for day in worst_days:
                        st.write(f"- {day}")
                else:
                    st.write("No severe weather conditions in the forecast.")

                query = f"Give me some recommendations for my {crop} crop. My city name is {city} and the weather is {weather} and Temperature is {temp}ºF"
                response = st.session_state.conversation.run(query)
                st.session_state.chat_history.append(("User", query))
                st.session_state.chat_history.append(("Assistant", response))
if st.session_state.chat_history:
    st.subheader("💬 Chat with the Crop Assistant")

    # Chat history display
    for speaker, message in st.session_state.chat_history:
        with st.chat_message("user" if speaker == "User" else "assistant"):
            st.markdown(message)

    # Chat input field
    follow_up = st.chat_input("Type your follow-up question here:")

    if follow_up:  # This will trigger when user hits enter
        if not groq_api_key:
            st.warning("OpenAI key is missing.")
        else:
            # Show user's message
            st.session_state.chat_history.append(("User", follow_up))
            with st.chat_message("user"):
                st.markdown(follow_up)

            # Run AI response
            reply = st.session_state.conversation.run(follow_up)
            st.session_state.chat_history.append(("Assistant", reply))

            with st.chat_message("assistant"):
                st.markdown(reply)

            st.rerun()

