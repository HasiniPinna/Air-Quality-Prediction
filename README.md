# 🌤️ AQI Vision — Intelligent Air Quality Prediction

An interactive web application that predicts Air Quality Index (AQI) using a Machine Learning model trained on real-world pollution data, combined with **live Hyderabad air quality data** fetched in real time.

Built with **Python**, **Streamlit**, and **Random Forest Regressor**.

---

## 🚀 Live Demo
> Run locally using the steps below ⬇️

---

## 📸 Features

| Feature | Description |
|---|---|
| 🌍 Live AQI Dashboard | Fetches real-time PM2.5, PM10, NO2, CO & AQI for Hyderabad |
| 📈 Live Line Chart | Hourly pollutant trends for the last 3 days |
| 🎯 ML AQI Predictor | Enter pollution values manually → get instant AQI prediction |
| 📊 Dataset Visualizer | Upload your own CSV and explore pollution trends |
| 🚨 Health Status Alerts | Color-coded warnings based on AQI severity level |

---

## 🧠 Machine Learning Model

- **Algorithm:** Random Forest Regressor
- **Input Features:** PM2.5, PM10, NO2, CO
- **Target Variable:** AQI
- **Library:** scikit-learn

The model is trained on the uploaded CSV dataset and predicts AQI based on manually entered pollutant values via interactive sliders.

---

## 📊 Datasets Used

### 1. `city_day.csv`
- City-wise daily air quality readings across India
- Columns: City, Date, PM2.5, PM10, NO2, CO, AQI, AQI_Bucket
- Source: [Kaggle — Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

### 2. `stations.csv`
- Details of air quality monitoring stations across India

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/HasiniPinna/Air-Quality-Prediction.git
cd Air-Quality-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements
streamlit
pandas
scikit-learn
requests
plotly

---

## 🏷️ AQI Health Categories

| AQI Range | Category | Meaning |
|---|---|---|
| 0 – 50 | 🟢 Good | Air is clean, safe for all |
| 51 – 100 | 🟡 Moderate | Acceptable; sensitive groups should take care |
| 101 – 150 | 🟠 Unhealthy for Sensitive Groups | Reduce outdoor activity if sensitive |
| 151 – 200 | 🔴 Unhealthy | Everyone may experience health effects |
| 200+ | 🚨 Severe / Hazardous | Avoid all outdoor activity, wear a mask |

---

## 👩‍💻 About the Developer

**Hasini Pinna**  
B.Tech Computer Science & Engineering (Data Science)  
Malla Reddy Engineering College for Women, Hyderabad  

---

> ⭐ If you found this project helpful, give it a star!
