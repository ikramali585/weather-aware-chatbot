
---

## 🌾 Crop Chat Assistant

A conversational Streamlit application that provides intelligent crop recommendations based on the weather and your crop type. It uses **Meta's LLaMA 3.2 model** via **Groq API** for chat interaction and **OpenWeatherMap API** for weather data.

---

### 🚀 Features

* Natural language conversation with Groq's blazing-fast **LLaMA 3.2**
* Weather-based crop recommendations using real-time data
* Forecast analysis to detect extreme weather conditions
* Persistent chat memory for smooth follow-up questions
* Clean, modern chat interface with Streamlit

---

### 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/crop-chat-assistant.git
cd crop-chat-assistant
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Create a `.env` file**

In the project root directory, create a `.env` file and add your API keys like this:

```env
GROQ_API_KEY=your_groq_api_key
WEATHER_API_KEY=your_openweathermap_api_key
```

> ⚠️ **Important:** Never share your `.env` file publicly. It contains sensitive credentials.

5. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

### 📂 Project Structure

```
crop-chat-assistant/
│
├── app.py              # Main Streamlit app
├── .env                # Your API keys
├── requirements.txt    # Dependencies
└── README.md           
```

---

### 📦 Technologies Used

* [Streamlit](https://streamlit.io/)
* [LangChain](https://www.langchain.com/)
* [Groq API](https://console.groq.com/)
* [Meta LLaMA 3.2 Model](https://ai.meta.com/llama/)
* [OpenWeatherMap API](https://openweathermap.org/)

---

### 🔐 API Setup

* **Groq API Key**
  Get your key from [console.groq.com](https://console.groq.com/)

* **OpenWeatherMap API Key**
  Register at [openweathermap.org](https://openweathermap.org/api)

Once you get the keys, paste them into the `.env` file as described above.

---

### 🧪 Coming Soon

* Crop disease detection via image upload
* Export chat as PDF or CSV
* Offline mode for local weather and model fallback

---

### 🤝 Contributing

Pull requests, feature suggestions, and issue reports are welcome!

---

### 📜 License

This project is licensed under the MIT License.

---
