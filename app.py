import requests
import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

api_key = ''  # weather API Key
os.environ["OPENAI_API_KEY"]=''  # Open AI key

crop_bot = ChatOpenAI()
conversation = ConversationChain(llm=crop_bot)


crop = input('Enter your crop\'s name: ')
user_city = input("Enter city: ")

weather_data = requests.get(
    f"https://api.openweathermap.org/data/2.5/weather?q={user_city}&units=imperial&APPID={api_key}")


if weather_data.json()['cod'] == '404':
    print("No City Found")
else:
    weather = weather_data.json()['weather'][0]['main']
    temp = round(weather_data.json()['main']['temp'])

def filter_data(data):
    unique_dates = set()
    filtered_data = []
    for entry in data['list']:
        date = entry['dt_txt'][0:-9]
        if date not in unique_dates:
            unique_dates.add(date)
            filtered_data.append(entry)
    return filtered_data

def check_weather_forecast(city):
    ndays = 40
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&cnt={ndays}&appid={api_key}"
    response = requests.get(url)
    if weather_data.json()['cod'] == '404':
        print("No City Found")
    # Parse the JSON response
    else:
        data = response.json()
        filtered_data = filter_data(data)
        rain_threshold_mm = 1.6 # Heavy rain threshold in mm
        wind_speed_threshold_mph = 20  # Strong winds threshold in mph
        high_temperature_threshold_celsius = 35  # Extreme heat threshold in °C
        low_temperature_threshold_celsius = 0  # Extreme cold threshold in °C

        worst_weather_days = []

        # Iterate through the forecast data
        for day in filtered_data:
            date = day['dt_txt']
            rain = day.get('rain', {}).get('3h', 0)
            wind_speed = day['wind']['speed']
            temperature = day['main']['temp']

            # Check for heavy rain, strong winds, or extreme temperatures
            if rain >= rain_threshold_mm or wind_speed >= wind_speed_threshold_mph or temperature >= high_temperature_threshold_celsius or temperature <= low_temperature_threshold_celsius:
                worst_weather_days.append(date)

        return worst_weather_days
    return

worst_weather = check_weather_forecast(user_city)
print(worst_weather)
querry = f'Give me some recommendations for my {crop} crop. My city name is {user_city} and the weather is {weather} and Temperature is {temp}ºF'
print(querry)
# info = conversation.run(querry)
# print(info)
