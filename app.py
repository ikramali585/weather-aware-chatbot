import streamlit as st
import os
import requests
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Crop Advisor", layout="wide")


# =========================
# INITIAL SETUP
# =========================
def load_env_vars():
    load_dotenv()

    secrets_dict = {}
    api_keys = {}
    try:
        # Accessing st.secrets directly can raise if secrets.toml is missing.
        secrets_dict = dict(st.secrets)
        maybe_api_keys = secrets_dict.get("api_keys", {})
        if isinstance(maybe_api_keys, dict):
            api_keys = maybe_api_keys
    except Exception:
        # No Streamlit secrets file configured; fall back to .env.
        pass

    # Prefer Streamlit secrets (nested or flat), then fall back to .env.
    groq_api_key = (
        api_keys.get("GROQ_API_KEY")
        or secrets_dict.get("GROQ_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )
    weather_api_key = (
        api_keys.get("WEATHER_API_KEY")
        or secrets_dict.get("WEATHER_API_KEY")
        or os.getenv("WEATHER_API_KEY")
    )

    return weather_api_key, groq_api_key


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
    try:
        response = requests.get(url, timeout=15)
        data = response.json()
    except requests.RequestException:
        return None, "Weather service is unreachable. Please try again."
    except ValueError:
        return None, "Weather service returned an invalid response."

    if response.status_code != 200:
        return None, data.get("message", "Unable to fetch weather forecast.")
    if not isinstance(data.get("list"), list):
        return None, "Unexpected forecast data format."

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
    st.markdown(
        '<h1><i class="fa-solid fa-cloud-sun" style="margin-right:0.38rem;"></i>Weather-aware Crop Advisor</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
        <style>
        .stApp {
            background: #f6f8f4;
        }
        .block-container {
            margin-top: 50px !important;
            padding-top: 1.3rem !important;
            padding-bottom: 1.4rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
            max-width: 1200px;
        }
        .stApp h1 {
            margin-top: 20px !important;
            margin-bottom: 0.85rem !important;
            background: linear-gradient(90deg, #1f5f33 0%, #2f8f47 52%, #53b86a 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }
        div[data-testid="stTextInput"] > div > div > input {
            border-radius: 12px;
            border: 1px solid #cfdcc5;
            background: #ffffff;
            padding: 0.52rem 0.7rem;
            transition: outline-color 0.15s ease, box-shadow 0.15s ease;
        }
        /* Streamlit/BaseWeb wrappers can add default focus ring; normalize and unify */
        div[data-testid="stTextInput"] div[data-baseweb="input"] {
            border-radius: 12px !important;
            border-color: #cfdcc5 !important;
            box-shadow: none !important;
        }
        div[data-testid="stChatInput"] {
            margin-top: 0.45rem;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }
        div[data-testid="stChatInput"] > div {
            border: 1px solid #d2dfc8 !important;
            box-shadow: none !important;
            outline: none !important;
            border-radius: 12px !important;
        }
        div[data-testid="stChatInput"] > div:focus-within {
            border: 1px solid #d2dfc8 !important;
            box-shadow: none !important;
            outline: 2px solid #2f6f3e !important;
            outline-offset: 1px !important;
            border-radius: 12px !important;
        }
        div[data-testid="stChatInput"] div[data-baseweb="textarea"] {
            border-radius: 12px !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
            transition: outline-color 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stChatInput"] textarea {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
            background: transparent !important;
        }
        div[data-testid="stChatInput"] button {
            background: #2f6f3e !important;
            border: 1px solid #265c33 !important;
            color: #ffffff !important;
        }
        div[data-testid="stChatInput"] button:hover {
            background: #265c33 !important;
        }
        div[data-testid="stChatInput"] button svg {
            fill: #ffffff !important;
            stroke: #ffffff !important;
        }
        div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
            border-color: #cfdcc5 !important;
            box-shadow: none !important;
            outline: 2px solid #2f6f3e !important;
            outline-offset: 1px !important;
        }
        div[data-testid="stChatInput"] div[data-baseweb="textarea"]:focus-within {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }
        div[data-testid="stTextInput"] > div > div > input:focus,
        div[data-testid="stTextInput"] > div > div > input:focus-visible,
        div[data-testid="stButton"] > button:focus,
        div[data-testid="stButton"] > button:focus-visible {
            border-color: #cfdcc5 !important;
            box-shadow: none !important;
            outline: 2px solid #2f6f3e !important;
            outline-offset: 1px !important;
        }
        div[data-testid="stButton"] > button {
            border-radius: 12px;
            border: none;
            background: #2f6f3e;
            color: white;
            font-weight: 600;
            padding: 0.52rem 1rem;
            transition: background 0.15s ease, outline-color 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stButton"] > button:hover {
            background: #265c33;
        }
        .wx-icon {
            color: #4f6647;
            margin-right: 0.35rem;
            width: 16px;
            text-align: center;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
        st.markdown("""
        <div style="background: #f4f9f1;
                    border: 1px solid #dcead4;
                    border-radius: 14px;
                    padding: 0.85rem 0.95rem;
                    margin: 0 0 0.7rem 0;">
            <h3 style="margin: 0; color: #000; font-size: 1.3rem;"><i class="fa-solid fa-sliders" style="margin-right:0.34rem;"></i>Input Information</h3>
            <p style="margin: 0.28rem 0 0 0; color: #56704f; font-size: 0.88rem;">
                Enter crop and city to get your weather-aware recommendation.
            </p>
        </div>
        """, unsafe_allow_html=True)
        crop = st.text_input("Enter your crop name:")
        city = st.text_input("Enter your city:")
        get_reco = st.button("Get Initial Recommendation")
        
        # Display weather data if available
        if st.session_state.weather_data:
            st.markdown("""
            <div style="background: #f4f9f1;
                        border: 1px solid #dcead4;
                        border-radius: 14px;
                        padding: 0.8rem 0.95rem;
                        margin: 0.2rem 0 0.7rem 0;">
                <h3 style="margin: 0; color: #000; font-size: 1.3rem;"><i class="fa-solid fa-cloud-sun" style="margin-right:0.34rem;"></i>Current Weather Data</h3>
            </div>
            """, unsafe_allow_html=True)
            weather_info = st.session_state.weather_data
            forecast_line = ""
            if weather_info.get('forecast_avg_temp') != 'N/A':
                forecast_line = f'<i class="fa-solid fa-chart-line wx-icon"></i><strong>24h Forecast:</strong> Avg Temp: {weather_info["forecast_avg_temp"]}°C &nbsp;|&nbsp; Avg Humidity: {weather_info["forecast_avg_humidity"]}% &nbsp;|&nbsp; Rain: {weather_info["forecast_rain"]}mm'
            forecast_line_block = f'<div style="margin-top: 0.34rem;">{forecast_line}</div>' if forecast_line else ""

            st.markdown(f"""
            <div style="background: linear-gradient(180deg, #f8fcf6 0%, #f1f8ed 100%);
                        border: 1px solid #d8e7cf;
                        border-radius: 14px;
                        padding: 0.9rem 1rem;
                        margin: 0.35rem 0 0.5rem 0;
                        color: #2e4829;
                        font-size: 0.92rem;
                        line-height: 1.55;
                        box-shadow: 0 3px 10px rgba(37, 73, 31, 0.06);">
                <div style="display: flex; align-items: center; gap: 0.42rem; margin-bottom: 0.28rem;">
                    <i class="fa-solid fa-cloud-sun wx-icon" style="font-size: 0.85rem;"></i>
                    <div style="font-weight: 700; font-size:20px">Weather in {st.session_state.current_city}</div>
                </div>
                <div style="margin-bottom: 0.45rem; color: #4f6647;">{weather_info['condition']} ({weather_info['description']})</div>
                <div style="background: #ffffff; border: 1px solid #e3eedf; border-radius: 10px; padding: 0.58rem 0.7rem;">
                    <div style="margin-bottom: 0.3rem;"><i class="fa-solid fa-temperature-three-quarters wx-icon"></i><strong>Temperature:</strong> {weather_info['temp_c']}°C ({weather_info['temp_f']}°F)</div>
                    <div style="margin-bottom: 0.3rem;"><i class="fa-solid fa-droplet wx-icon"></i><strong>Humidity:</strong> {weather_info['humidity']}% &nbsp;|&nbsp; <i class="fa-solid fa-wind wx-icon"></i><strong>Wind:</strong> {weather_info['wind_speed']} m/s</div>
                    <div style="margin-bottom: 0.3rem;"><i class="fa-solid fa-gauge-high wx-icon"></i><strong>Pressure:</strong> {weather_info['pressure']} hPa &nbsp;|&nbsp; <i class="fa-solid fa-eye wx-icon"></i><strong>Visibility:</strong> {weather_info['visibility']}m</div>
                    {forecast_line_block}
                </div>
            </div>
            """, unsafe_allow_html=True)
        if get_reco:
            if not weather_api_key:
                st.error("Please set your Weather API key in .env file.")
                return

            if not crop or not city:
                st.warning("Please fill in both crop name and city.")
                return

            # Fetch current weather
            try:
                current_resp = requests.get(
                    f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&APPID={weather_api_key}",
                    timeout=15,
                )
                weather_data = current_resp.json()
            except requests.RequestException:
                st.error("Weather service is unreachable. Please try again.")
                return
            except ValueError:
                st.error("Weather service returned an invalid response.")
                return

            if current_resp.status_code != 200:
                st.error(weather_data.get("message", "Unable to fetch current weather data."))
                return

            if (
                not isinstance(weather_data.get("weather"), list)
                or not weather_data["weather"]
                or "main" not in weather_data
                or "wind" not in weather_data
            ):
                st.error("Unexpected current weather data format.")
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
            forecast_resp = None
            try:
                forecast_resp = requests.get(
                    f"https://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&APPID={weather_api_key}",
                    timeout=15,
                )
                forecast_data = forecast_resp.json()
            except requests.RequestException:
                forecast_data = {}
            except ValueError:
                forecast_data = {}
            
            # Initialize forecast variables
            avg_temp = "N/A"
            avg_humidity = "N/A"
            total_rain = "N/A"
            
            if (
                forecast_resp is not None
                and forecast_resp.status_code == 200
                and isinstance(forecast_data.get("list"), list)
            ):
                # Analyze forecast trends
                forecast_list = forecast_data["list"][:8]  # Next 24 hours (8 x 3-hour intervals)
                if forecast_list:
                    valid_main = [item["main"] for item in forecast_list if "main" in item]
                    if valid_main:
                        avg_temp = round(
                            sum(item["temp"] for item in valid_main) / len(valid_main)
                        )
                        avg_humidity = round(
                            sum(item["humidity"] for item in valid_main) / len(valid_main)
                        )
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
                st.markdown('<p><i class="fa-solid fa-cloud-showers-heavy" style="margin-right:0.35rem;"></i><strong>Worst Weather Days:</strong></p>', unsafe_allow_html=True)
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
            <div style="background: #edf4e8;
                        border: 1px solid #d1dfc7;
                        padding: 0.9rem 1rem;
                        border-radius: 12px;
                        text-align: center;
                        margin: 0.75rem 0 0.9rem 0;">
                <p style="color: #244723; margin: 0; font-weight: 650; font-size: 0.98rem;">
                    <i class="fa-solid fa-circle-check" style="margin-right:0.35rem;"></i>Recommendation added to chatbot panel →
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
        <div style="background: #2f6f3e;
                    padding: 0.9rem 1rem;
                    border-radius: 14px;
                    margin-bottom: 0.9rem;
                    text-align: center;">
            <h3 style="color: #f6faef; margin: 0; font-size: 1.2rem;">
                <i class="fa-solid fa-seedling" style="margin-right:0.35rem;"></i>AI Crop Assistant
            </h3>
            <p style="color: #e6f0dc; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
                Your intelligent farming companion
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Chat messages container with enhanced styling
        if st.session_state.chat_history:
            st.markdown('<h3 style="margin: 0.2rem 0 0.6rem 0; color: #000; font-size: 1.3rem;"><i class="fa-solid fa-comments" style="margin-right:0.35rem;"></i>Conversation History</h3>', unsafe_allow_html=True)
            
            # Create a scrollable container for chat messages
            chat_container = st.container()
            with chat_container:
                for speaker, message in st.session_state.chat_history:
                    if speaker == "User":
                        # Extract first line for display
                        first_line = message.split('\n')[0].strip()
                        # Enhanced user message styling
                        st.markdown(f"""
                        <div style="background: #2f6f3e;
                                    border: 1px solid #3a7d4b;
                                    padding: 0.8rem 0.9rem;
                                    border-radius: 14px 14px 6px 14px;
                                    margin: 0.5rem 0;
                                    color: white;
                                    box-shadow: none;">
                            <div style="display: flex; align-items: center; margin-bottom: 0.28rem;">
                                <i class="fa-solid fa-user" style="font-size: 0.9rem; margin-right: 0.45rem;"></i>
                                <strong>You asked:</strong>
                            </div>
                            <div style="font-size: 0.93rem;">
                                {first_line}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Enhanced assistant message styling
                        st.markdown(f"""
                        <div style="background: #f7fbf5;
                                    border: 1px solid #e6efe2;
                                    padding: 1rem 1.05rem;
                                    border-radius: 16px 16px 16px 8px;
                                    margin: 0.6rem 0;
                                    color: #263424;
                                    box-shadow: 0 6px 18px rgba(39, 71, 33, 0.10);">
                            <div style="display: flex; align-items: center; margin-bottom: 0.45rem;">
                                <span style="font-size: 1rem; margin-right: 0.5rem; width: 1.55rem; height: 1.55rem; border-radius: 50%; background: #eaf5e3; border: 1px solid #d0e3c6; display: inline-flex; align-items: center; justify-content: center;"><i class="fa-solid fa-robot" style="font-size: 0.78rem; color: #3c5a35;"></i></span>
                                <strong style="font-size: 0.78rem; color: #30502a; letter-spacing: 0.6px; text-transform: uppercase; background: #ecf7e6; border: 1px solid #d0e4c3; border-radius: 999px; padding: 0.2rem 0.55rem;">AI Assistant</strong>
                            </div>
                            <div style="font-size: 0.97rem; line-height: 1.6;">
                                {message}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            # Welcome message when no chat history
            st.markdown("""
            <div style="background: #eef5e8;
                        border: 1px solid #d2dfc8;
                        padding: 1.3rem;
                        border-radius: 14px;
                        text-align: center;
                        margin: 0.75rem 0;">
                <h3 style="color: #000; margin: 0 0 0.9rem 0;"><i class="fa-solid fa-seedling" style="margin-right:0.35rem;"></i>Welcome to AI Crop Assistant!</h3>
                <p style="color: #4f6148; margin: 0; font-size: 0.95rem;">
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
            <div style="background: #f6f2e8;
                        border: 1px solid #e3dbc9;
                        padding: 0.9rem;
                        border-radius: 12px;
                        text-align: center;
                        margin: 0.75rem 0;">
                <p style="color: #6b5530; margin: 0; font-weight: 600;">
                    <i class="fa-solid fa-lock" style="margin-right:0.35rem;"></i>Chat is locked - Get your initial recommendation first!
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.chat_input("Ask a follow-up question...", disabled=True)
        else:
            # Active chat input with enhanced styling
            st.markdown("""
            <div style="background: #eef5e8;
                        border: 1px solid #d2dfc8;
                        padding: 0.82rem;
                        border-radius: 12px;
                        text-align: center;
                        margin-bottom: 0.5rem;">
                <p style="color: #2f4d2a; margin: 0; font-weight: 600;">
                    <i class="fa-solid fa-comment-dots" style="margin-right:0.35rem;"></i>Ready to chat! Ask me anything about your crops
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



