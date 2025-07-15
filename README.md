
# 🌍 Clima AI - Intelligent Weather Forecasting & Climate Insights

Clima AI is a powerful Streamlit-based weather forecasting application that combines **live weather updates**, **ML-powered prediction**, and **climate trend analysis** using open-source weather APIs and machine learning.

> "Predict Tomorrow, Understand Today."

---

## 🚀 Features

- 🔍 Live weather data by city (via WeatherAPI)
- 🧠 ML-based Max Temperature Prediction (Random Forest)
- 📊 Historical Climate Trends (Temp, Rain, Wind, Humidity)
- ⚠️ Z-Score-Based Anomaly Detection
- 💡 Smart Weather Recommendations
- 🔁 On-Demand Model Retraining
- 🌐 Two-City Weather Comparison
- ⬇️ Downloadable Cleaned Weather Dataset
- 📱 Mobile-Responsive Design

---

## 🖼️ Preview

<img src="assets/climaai_preview.png" width="800"/>

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-darkred?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-black?logo=plotly)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-orange?logo=scikit-learn)
![Open-Meteo](https://img.shields.io/badge/Open--Meteo-API-blue)
![WeatherAPI](https://img.shields.io/badge/WeatherAPI-REST-brightgreen)

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/clima-ai.git
cd clima-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a file `.streamlit/secrets.toml` and add your [WeatherAPI](https://www.weatherapi.com/) key:

```toml
WEATHERAPI_KEY = "your_api_key_here"
```

### 4. Run the App

```bash
streamlit run app.py
```

--

## 📦 Requirements

See [`requirements.txt`](requirements.txt)

---

## 📂 Project Structure

```
├── app.py                  # Main Streamlit app
├── assets/
│   └── climaai_preview.png  # Thumbnail preview
├── requirements.txt
└── README.md
```

---


