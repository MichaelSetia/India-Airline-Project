# ✈️ India Airline Price & Analytics Predictor

An end-to-end machine learning project that analyzes and predicts airline ticket prices and travel details for domestic flights in India. It covers the full pipeline: exploratory data analysis (EDA), data cleaning, model training, and deployment as an interactive **Streamlit** web app.

<img width="1512" height="788" alt="image" src="https://github.com/user-attachments/assets/47d4c0a4-0a20-48b1-9362-077cc11c3ff1" />

# Link Website
https://india-airline-project-ypnlfweykkwitsrmjbgipw.streamlit.app/
---

## 📌 Overview

- Explore and clean airline data in a Jupyter Notebook
- Engineer features and encode/scale them with reusable preprocessing artifacts
- Train and compress a Random Forest model for fast inference
- Serve predictions through a simple web interface built with Streamlit

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Web app | Streamlit |
| Data & ML | Python, Pandas, NumPy, Scikit-learn |
| Experimentation | Jupyter Notebook |
| Serialization | Joblib / Pickle |

---

## 📁 Repository Structure

```text
India-Airline-Project/
├── app.py                        # Streamlit web application
├── preprocessing.py              # Data cleaning & feature engineering
├── calculation.py                # Supporting calculation logic
├── compress_model.py             # Reduces model file size
├── Airline_Project_ipynb.ipynb   # EDA, data cleaning & model training
├── arrival_lookup.csv            # Reference data for arrival time calculation
├── duration_lookup.csv           # Reference data for flight duration estimation
├── rf_default_model_small.pkl    # Trained & compressed Random Forest model
├── scaler.pkl                    # Fitted scaler
├── feature.pkl                   # Feature column list
├── le_dict.pkl                   # Label encoders
├── ohe_dict.pkl                  # One-hot encoders
├── requirements.txt              # Python dependencies
└── .gitignore
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/MichaelSetia/India-Airline-Project.git
cd India-Airline-Project
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the App

```bash
streamlit run app.py
```

Streamlit will open the app in your browser, usually at `http://localhost:8501`.

---

## 🧠 How It Works

1. **Input:** The user enters flight details in the Streamlit interface.
2. **Preprocessing:** `preprocessing.py` and `calculation.py` clean the input and derive features, using the lookup CSVs for duration and arrival values.
3. **Encoding & scaling:** The saved encoders (`le_dict.pkl`, `ohe_dict.pkl`), feature list (`feature.pkl`) and `scaler.pkl` make sure the input matches the training format.
4. **Prediction:** The pre-trained Random Forest model (`rf_default_model_small.pkl`) returns the predicted result, which is displayed in the app.

---

## 📓 Notebook

`Airline_Project_ipynb.ipynb` contains the full workflow: data exploration, cleaning, feature engineering, and model training. Open it with:

```bash
jupyter notebook Airline_Project_ipynb.ipynb
```

---

## 👤 Author

**Michael Setia** — [GitHub](https://github.com/MichaelSetia)
