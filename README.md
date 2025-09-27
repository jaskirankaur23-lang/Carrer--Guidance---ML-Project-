# Career Guidance System (Machine Learning)  

## 📌 Overview  
This project is a *Career Guidance System* built using *Machine Learning*.  
It predicts suitable career paths for students based on their input, interests, and skills.  
The system uses natural language processing (NLP) and trained ML models to suggest the best career options.  

---

## 📂 Project Files  

- *career_file.csv* → Dataset containing career-related training data.  
- *train_model.py* → Python script to train the ML model.  
- *career_model.h5 / model.h5* → Saved trained models.  
- *encoder.pkl / label_encoder.pkl* → Encoders for categorical data.  
- *tfidf_vectorizer.pkl / vectorizer.pkl* → Vectorizers for text feature extraction.  

---

## ⚙ Features  

- Predicts career paths using *Machine Learning* models.  
- Uses *TF-IDF Vectorizer* for text processing.  
- Encodes categorical inputs for better accuracy.  
- Provides a base to build a *Career Guidance Web or Mobile Application*.  

---

## 🚀 How to Run  

### 1. Clone the repository  
```bash
git clone https://github.com/jaskirankaur23-lang/career-guidance-ml.git
cd career-guidance-ml

##Code
# ---------- IMPORT LIBRARIES ----------

import os  # For checking files and handling file paths
import pickle  # For saving/loading models and objects
import pandas as pd  # For data processing using DataFrames
import numpy as np  # For numerical operations
import spacy  # For Natural Language Processing
from sklearn.model_selection import train_test_split  # For splitting data into training/testing sets
from sklearn.preprocessing import LabelEncoder  # For converting career labels into numbers
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text into numbers
from tensorflow.keras.models import Sequential, load_model  # For creating/loading deep learning models
from tensorflow.keras.layers import Dense  # Dense = fully connected layer in neural network
from tensorflow.keras.utils import to_categorical  # Converts labels to one-hot encoding
import streamlit as st  # For creating the web interface

# ---------- LOAD NLP MODEL ----------

nlp = spacy.load("en_core_web_sm")  # Load the small English NLP model

# ---------- LOAD DATA ----------

data = pd.read_csv("career_file.csv", encoding="latin1")  # Load CSV file into DataFrame

# Select only the useful columns from dataset
columns_we_use = [
    'FIELD', 'INTEREST AREAS', 'REQUIRED SKILLS', 'DEGREES', 'CERTIFICATIONS',
    'TOOLS/TECHNOLOGIES', 'PROJECT EXAMPLES', 'ENTRY LEVEL JOB TITLE',
    'HIGHER OPPORTUNITIES', 'TOP COMPANIES', 'LEARNING RESOURCES', 'ROADMAP'
]

# Combine all columns of text into one column for ML model training
combined_text = data[columns_we_use].astype(str).agg(' '.join, axis=1)

# ---------- LOAD OR TRAIN MODEL ----------

# Check if pre-trained model files exist
if os.path.exists("model.h5") and os.path.exists("vectorizer.pkl") and os.path.exists("encoder.pkl"):
    model = load_model("model.h5")  # Load trained model
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)  # Load vectorizer
    with open("encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)  # Load label encoder
else:
    st.write("⚠ Model files not found. Training model now...")  # Show message if model needs to be trained

    # Encode target labels into numbers, then convert to one-hot encoding
    label_encoder = LabelEncoder()
    y = to_categorical(label_encoder.fit_transform(data['CAREER']))

    # Convert combined text to numerical format
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(combined_text).toarray()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create deep learning model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),  # Input + first hidden layer
        Dense(64, activation='relu'),  # Second hidden layer
        Dense(y_train.shape[1], activation='softmax')  # Output layer with softmax for classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Show loading spinner while training
    with st.spinner("Training the model..."):
        model.fit(X_train, y_train, epochs=16, batch_size=8, verbose=0)

    # Save the trained model and tools
    model.save("model.h5")
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

# ---------- UI STYLING ----------

st.set_page_config(page_title="Career Guidance", layout="centered")  # Page title and layout

# Add custom CSS styling to make UI beautiful
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');

    body {
        background-color: #3b0a58;
        background-image: linear-gradient(120deg, #4c0b85, #2c003e);
        color: white;
        font-family: 'Poppins', sans-serif;
    }

    h1 {
        text-align: center;
        font-size: 14em;
        font-weight: 900;
        margin-bottom: 20px;
        color: #ffe600;
        text-shadow: 3px 3px #000;
    }

    h2 {
        text-align: center;
        font-size: 2.5em;
        margin-top: 0;
        color: #ffffff;
    }

    .quote-box {
        text-align: center;
        font-size: 1.3em;
        color: #ffffff;
        margin-bottom: 40px;
    }

    .stTextInput > div > label {
        font-weight: bold;
        font-size: 1.6rem !important;
        color: #ffffff;
    }

    .stButton>button {
        background-color: #ffe600;
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75em 1.5em;
        font-size: 1.3em;
        width: 100%;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #fff300;
        transform: scale(1.03);
    }

    .career-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 30px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------

st.markdown("<h1>🚀 Career Guidance</h1>", unsafe_allow_html=True)
st.markdown("<h2>Looking for the <span style='color:#ffe600;'>RIGHT CAREER PATH?</span></h2>", unsafe_allow_html=True)

# ---------- QUOTES ----------

st.markdown("""
    <div class="quote-box">
        <p>“Choose a job you love, and you’ll never have to work a day in your life.”</p>
        <p>“Your future depends on what you do today. Let’s shape it together!”</p>
    </div>
""", unsafe_allow_html=True)

# ---------- INPUT SECTION ----------

st.markdown("<div class='box'>", unsafe_allow_html=True)

# Create two side-by-side columns for input questions
col1, col2 = st.columns(2)
with col1:
    q1 = st.text_input("📚 What was your stream in 12th?")
    q2 = st.text_input("🧠 what is your favorite subject in 12th?")
    q3 = st.text_input("🌐 what is your desired working field?")
with col2:
    q4 = st.text_input("🎨 what are your favourite activities or tasks?")
    q5 = st.text_input("🏢 what work environment you preffered?")
    q6 = st.text_input("🔍 which topics you'd love to explore?")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- BUTTON & RECOMMENDATION LOGIC ----------

if st.button("🔎 Find My Career"):
    interest_inputs = [q1, q2, q4, q5, q6]  # Exclude q3 (working field) from interest matching
    interest_words = set(" ".join(interest_inputs).lower().split())  # Create set of interest keywords
    field_words = set(str(q3).lower().split())  # Create set of working field keywords

    results = []  # To store matched careers
    for i, row in data.iterrows():
        row_field = set(str(row["FIELD"]).lower().replace(",", " ").split())
        row_interest = set(str(row["INTEREST AREAS"]).lower().replace(",", " ").split())

        field_match = bool(field_words & row_field)  # Check if fields match
        interest_match_count = sum(bool(set(ans.lower().split()) & row_interest) for ans in interest_inputs)

        if field_match and interest_match_count >= 2:  # If match is strong
            results.append((i, interest_match_count))

    # ---------- IF MATCH FOUND ----------

    if results:
        st.success("✨ Careers matching your passion and preferences:")
        for i, score in results:
            row = data.iloc[i]
            st.markdown(f"<div class='career-card'><h3>🔹 {row['CAREER']} ({row['FIELD']})</h3>", unsafe_allow_html=True)
            with st.expander("📋 Career Details"):
                for col in columns_we_use:
                    value = str(row[col])
                    if value.strip():
                        st.markdown(f"- *{col}:* {value}")
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- IF NO MATCH FOUND: USE AI ----------
    
    else:
        st.warning("✨ Careers matching your passion and preferences:")

        user_input = " ".join([q1, q2, q3, q4, q5, q6])  # Combine all user answers
        vec = vectorizer.transform([user_input]).toarray()  # Convert to vector
        pred = model.predict(vec)  # Predict using ML model
        pred_index = np.argmax(pred)  # Get the highest probability index
        pred_career = label_encoder.inverse_transform([pred_index])[0]  # Convert back to label
        row = data[data["CAREER"] == pred_career].iloc[0]  # Get row info

        st.markdown(f"<div class='career-card'><h3>🎯 AI Predicted Career: {pred_career} ({row['FIELD']})</h3>", unsafe_allow_html=True)
        with st.expander("📋 Career Details"):
            for col in columns_we_use:
                val = str(row[col])
                if val.strip():
                    st.markdown(f"- *{col}:* {val}")
        st.markdown("</div>", unsafe_allow_html=True)

