# 🚦 GreenCommute – Smart Traffic & AQI Analysis System

## Overview

GreenCommute is a computer vision–based system that analyzes CCTV road footage to monitor traffic patterns, detect violations, and estimate environmental impact.

The system uses OpenCV-based video processing to:

* Count vehicles in real-time
* Classify vehicles by type
* Estimate Air Quality Index (AQI) based on traffic density and vehicle categories
* Detect over-speeding vehicles
* Capture and store screenshots of violating vehicles

---

## ⚙️ Features

* **Vehicle Detection & Counting**
  Detects and counts vehicles from CCTV footage.

* **Vehicle Classification**
  Categorizes vehicles (cars, bikes, trucks, etc.).

* 🌫️ **AQI Estimation**
  Calculates approximate AQI using:

  * Number of vehicles
  * Type of vehicles
  * Traffic density factors

* **Over-speed Detection**
  Identifies vehicles exceeding speed thresholds.

* **Violation Capture**
  Automatically captures screenshots of over-speeding vehicles and stores them in a dedicated folder.

* **Visualization**
  Displays processed data using graphs and UI (via Streamlit).

---

## Tech Stack

* Python
* OpenCV
* Streamlit
* NumPy
* Matplotlib
* Pillow

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/GreenCommute.git
cd GreenCommute
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## Output

* Processed video stream
* Vehicle count statistics
* AQI estimation results
* Screenshots of over-speeding vehicles (saved locally)

---

## Future Improvements

* Integration with real-time CCTV feeds
* AI-based accurate AQI prediction models
* License plate recognition (ANPR)
* Cloud storage for violation records

---

## Use Case

* Smart city traffic monitoring
* Pollution tracking & analysis
* Law enforcement assistance
* Urban planning insights

---

