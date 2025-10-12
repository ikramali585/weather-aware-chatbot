import streamlit as st
import os
import requests
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="🌾 Crop Advisor", layout="wide")


# =========================
# INITIAL SETUP
# =========================
def load_env_vars():
    load_dotenv()
    return os.getenv("WEATHER_API_KEY"), os.getenv("GROQ_API_KEY")


def init_groq_conversation(groq_api_key: str):
    if not groq_api_key:
        st.warning("Please set Groq API key in the environment variables.")
        return None

    os.environ["GROQ_API_KEY"] = groq_api_key
    warnings.filterwarnings("ignore", message=".*ConversationChain.*")
    warnings.filterwarnings("ignore", message=".*Chain.run.*")

    crop_bot = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
    memory = ConversationBufferMemory()
    return ConversationChain(llm=crop_bot, memory=memory)


def call_conversation(conversation_obj, query: str) -> str:
    """Compatibility helper to call LangChain conversation."""
    try:
        return conversation_obj.invoke(query)
    except Exception:
        try:
            return conversation_obj.invoke({"input": query})
        except Exception:
            return conversation_obj.run(query)


# =========================
# WEATHER HELPERS
# =========================
def filter_data(data):
    unique_dates = set()
    filtered_data = []
    for entry in data["list"]:
        date = entry["dt_txt"][:-9]
        if date not in unique_dates:
            unique_dates.add(date)
            filtered_data.append(entry)
    return filtered_data


def check_weather_forecast(city, api_key):
    ndays = 40
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&cnt={ndays}&appid={api_key}"
    response = requests.get(url)
    if response.status_code != 200 or response.json().get("cod") == "404":
        return None, "City not found"

    data = response.json()
    filtered_data = filter_data(data)

    rain_threshold = 1.6
    wind_threshold = 20
    high_temp = 35
    low_temp = 0

    worst_days = []
    for day in filtered_data:
        date = day["dt_txt"]
        rain = day.get("rain", {}).get("3h", 0)
        wind = day["wind"]["speed"]
        temp_c = day["main"]["temp"] - 273.15

        if (
            rain >= rain_threshold
            or wind >= wind_threshold
            or temp_c >= high_temp
            or temp_c <= low_temp
        ):
            worst_days.append(date)

    return worst_days, None


# =========================
# MAIN APP LAYOUT
# =========================
def main():
    st.title("🌦️ Weather-aware Crop Advisor")

    weather_api_key, groq_api_key = load_env_vars()
    if "conversation" not in st.session_state:
        st.session_state.conversation = init_groq_conversation(groq_api_key)
        st.session_state.chat_history = []
        # Flag to track whether the initial recommendation has been generated
        st.session_state.initial_reco_done = False
        # Store weather data persistently
        st.session_state.weather_data = None
        st.session_state.current_city = None
        st.session_state.current_crop = None

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    # -------- LEFT COLUMN: Inputs --------
    with col1:
        st.subheader("🧩 Input Information")
        crop = st.text_input("Enter your crop name:")
        city = st.text_input("Enter your city:")
        get_reco = st.button("Get Initial Recommendation")
        
        # Display weather data if available
        if st.session_state.weather_data:
            st.subheader("🌤️ Current Weather Data")
            weather_info = st.session_state.weather_data
            
            st.success(f"Weather in {st.session_state.current_city}: {weather_info['condition']} ({weather_info['description']})")
            st.info(f"🌡️ Temperature: {weather_info['temp_c']}°C ({weather_info['temp_f']}°F)")
            st.info(f"💧 Humidity: {weather_info['humidity']}% | 🌬️ Wind: {weather_info['wind_speed']} m/s")
            st.info(f"📊 Pressure: {weather_info['pressure']} hPa | 👁️ Visibility: {weather_info['visibility']}m")
            
            if weather_info.get('forecast_avg_temp') != 'N/A':
                st.info(f"📈 24h Forecast: Avg Temp: {weather_info['forecast_avg_temp']}°C | Avg Humidity: {weather_info['forecast_avg_humidity']}% | Rain: {weather_info['forecast_rain']}mm")

        if get_reco:
            if not weather_api_key:
                st.error("Please set your Weather API key in .env file.")
                return

            if not crop or not city:
                st.warning("Please fill in both crop name and city.")
                return

            # Fetch current weather
            weather_data = requests.get(
                f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&APPID={weather_api_key}"
            ).json()

            if weather_data.get("cod") == "404":
                st.error("City not found!")
                return

            # Extract comprehensive weather data
            weather_condition = weather_data["weather"][0]["main"]
            weather_description = weather_data["weather"][0]["description"]
            temp_celsius = round(weather_data["main"]["temp"])
            temp_fahrenheit = round(temp_celsius * 9/5 + 32)
            feels_like_c = round(weather_data["main"]["feels_like"])
            humidity = weather_data["main"]["humidity"]
            pressure = weather_data["main"]["pressure"]
            wind_speed = weather_data["wind"]["speed"]
            wind_direction = weather_data["wind"].get("deg", "N/A")
            visibility = weather_data.get("visibility", "N/A")
            uv_index = weather_data.get("uv", "N/A")
            
            # Get 5-day forecast for better analysis
            forecast_data = requests.get(
                f"https://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&APPID={weather_api_key}"
            ).json()
            
            # Initialize forecast variables
            avg_temp = "N/A"
            avg_humidity = "N/A"
            total_rain = "N/A"
            
            if forecast_data.get("cod") == "200":
                # Analyze forecast trends
                forecast_list = forecast_data["list"][:8]  # Next 24 hours (8 x 3-hour intervals)
                avg_temp = round(sum(item["main"]["temp"] for item in forecast_list) / len(forecast_list))
                avg_humidity = round(sum(item["main"]["humidity"] for item in forecast_list) / len(forecast_list))
                total_rain = sum(item.get("rain", {}).get("3h", 0) for item in forecast_list)
            
            # Store weather data in session state
            st.session_state.weather_data = {
                'condition': weather_condition,
                'description': weather_description,
                'temp_c': temp_celsius,
                'temp_f': temp_fahrenheit,
                'feels_like': feels_like_c,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'visibility': visibility,
                'uv_index': uv_index,
                'forecast_avg_temp': avg_temp,
                'forecast_avg_humidity': avg_humidity,
                'forecast_rain': total_rain
            }
            st.session_state.current_city = city
            st.session_state.current_crop = crop

            worst_days, error = check_weather_forecast(city, weather_api_key)
            if error:
                st.error(error)
            else:
                st.write("🌧️ **Worst Weather Days:**")
                if worst_days:
                    for d in worst_days:
                        st.write(f"- {d}")
                else:
                    st.write("No severe weather in forecast.")

            # Create comprehensive prompt for crop recommendations
            query = f"""
            As an expert agricultural advisor, please provide brief recommendations for growing {crop} in {city}. Consider the following current conditions and factors:

            CURRENT WEATHER CONDITIONS:
            - Weather: {weather_condition} ({weather_description})
            - Temperature: {temp_celsius}°C ({temp_fahrenheit}°F), Feels like: {feels_like_c}°C
            - Humidity: {humidity}%
            - Atmospheric Pressure: {pressure} hPa
            - Wind Speed: {wind_speed} m/s, Direction: {wind_direction}°
            - Visibility: {visibility}m
            - UV Index: {uv_index}

            FORECAST TRENDS (24h):
            - Average Temperature: {round(avg_temp) if 'avg_temp' in locals() else 'N/A'}°C
            - Average Humidity: {round(avg_humidity) if 'avg_humidity' in locals() else 'N/A'}%
            - Expected Rainfall: {total_rain if 'total_rain' in locals() else 'N/A'}mm

            Please provide recommendations briefly considering:
            1. Optimal growing conditions for {crop}
            2. Current weather suitability and potential risks
            3. Seasonal timing and planting windows
            4. Soil preparation and irrigation needs
            5. Pest and disease management based on weather conditions
            6. Harvest timing considerations
            7. Any weather-related precautions or protective measures
            8. Alternative crops if current conditions are unfavorable

            Provide specific, actionable advice tailored to the current conditions in {city}.
            Provide the answer in a concise manner.
            """
            response = call_conversation(st.session_state.conversation, query)
            reply = response["response"] if isinstance(response, dict) else response

            st.session_state.chat_history.append(("User", query))
            st.session_state.chat_history.append(("Assistant", reply))
            
            # Enhanced success message
            st.markdown("""
            <div style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); 
                        padding: 1rem; 
                        border-radius: 10px; 
                        text-align: center;
                        margin: 1rem 0;">
                <p style="color: #2d5016; margin: 0; font-weight: 600; font-size: 1rem;">
                    ✅ Recommendation added to chatbot panel →
                </p>
            </div>
            """, unsafe_allow_html=True)
            # Mark that initial recommendation has been generated so chat input is enabled
            st.session_state.initial_reco_done = True
            # Rerun so the UI updates and enables the chat input
            st.rerun()

    # -------- RIGHT COLUMN: Chatbot --------
    with col2:
        # Enhanced chat header with attractive styling
        st.markdown("""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; 
                    border-radius: 10px; 
                    margin-bottom: 1rem;
                    text-align: center;">
            <h3 style="color: white; margin: 0; font-size: 1.2rem;">
                🌱🤖 AI Crop Assistant
            </h3>
            <p style="color: #e0e0e0; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                Your intelligent farming companion
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Chat messages container with enhanced styling
        if st.session_state.chat_history:
            st.markdown("### 💭 Conversation History")
            
            # Create a scrollable container for chat messages
            chat_container = st.container()
            with chat_container:
                for speaker, message in st.session_state.chat_history:
                    if speaker == "User":
                        # Extract first line for display
                        first_line = message.split('\n')[0].strip()
                        # Enhanced user message styling
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 0.8rem; 
                                    border-radius: 15px 15px 5px 15px; 
                                    margin: 0.5rem 0; 
                                    color: white;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="display: flex; align-items: center; margin-bottom: 0.3rem;">
                                <span style="font-size: 1.2rem; margin-right: 0.5rem;">👤</span>
                                <strong>You asked:</strong>
                            </div>
                            <div style="font-size: 0.95rem;">
                                {first_line}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Enhanced assistant message styling
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                    padding: 0.8rem; 
                                    border-radius: 15px 15px 15px 5px; 
                                    margin: 0.5rem 0; 
                                    color: white;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="display: flex; align-items: center; margin-bottom: 0.3rem;">
                                <span style="font-size: 1.2rem; margin-right: 0.5rem;">🌾</span>
                                <strong>AI Assistant replied:</strong>
                            </div>
                            <div style="font-size: 0.95rem; line-height: 1.4;">
                                {message}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            # Welcome message when no chat history
            st.markdown("""
            <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 2rem; 
                        border-radius: 15px; 
                        text-align: center;
                        margin: 1rem 0;">
                <h4 style="color: #333; margin: 0 0 1rem 0;">🌱 Welcome to AI Crop Assistant!</h4>
                <p style="color: #666; margin: 0; font-size: 0.95rem;">
                    Get personalized crop recommendations based on your location and weather conditions.
                    Start by getting your initial recommendation on the left!
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Enhanced chat input section
        st.markdown("---")
        
        if not st.session_state.get("initial_reco_done", False):
            # Disabled state with attractive styling
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                        padding: 1rem; 
                        border-radius: 10px; 
                        text-align: center;
                        margin: 1rem 0;">
                <p style="color: #8b4513; margin: 0; font-weight: 500;">
                    🔒 Chat is locked - Get your initial recommendation first!
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.chat_input("Ask a follow-up question...", disabled=True)
        else:
            # Active chat input with enhanced styling
            st.markdown("""
            <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 0.8rem; 
                        border-radius: 10px; 
                        text-align: center;
                        margin-bottom: 0.5rem;">
                <p style="color: #333; margin: 0; font-weight: 500;">
                    💬 Ready to chat! Ask me anything about your crops
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            follow_up = st.chat_input("Ask a follow-up question...")

            if follow_up:
                st.session_state.chat_history.append(("User", follow_up))
                reply = call_conversation(st.session_state.conversation, follow_up)
                reply_text = reply["response"] if isinstance(reply, dict) else reply
                st.session_state.chat_history.append(("Assistant", reply_text))
                st.rerun()


if __name__ == "__main__":
    main()
