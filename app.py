import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add project root to path so we can import recommender
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "ml"))
from recommender import generate_recommendations

# --- Paths ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "src", "ml", "models")

# --- Grade Labels ---
GRADE_MAP = {0: "Grade 0", 1: "Grade 1", 2: "Grade 2", 3: "Grade 3", 4: "Grade 4", 5: "Grade 5"}
CATEGORY_MAP = {0: "At-Risk", 1: "Below-Average", 2: "Average", 3: "Above-Average", 4: "High-Performing", 5: "Exceptional"}


@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
        features = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scale_cols.pkl"), "rb") as f:
        scale_cols = pickle.load(f)
    return model, features, scaler, scale_cols


def preprocess_raw_data(df, scaler, scale_cols):
    df = df.copy()

    if "student_id" in df.columns:
        df.drop(columns=["student_id"], inplace=True)

    if "final_grade" in df.columns:
        df.drop(columns=["final_grade"], inplace=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    binary_map = {"yes": 1, "no": 0}
    for col in ["internet_access", "extra_activities"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    travel_map = {"<15 min": 0, "15-30 min": 1, "30-60 min": 2, ">60 min": 3}
    if "travel_time" in df.columns and df["travel_time"].dtype == "object":
        df["travel_time"] = df["travel_time"].map(travel_map).fillna(0).astype(int)

    edu_map = {"no formal": 0, "high school": 1, "diploma": 2, "graduate": 3, "post graduate": 4, "phd": 5}
    if "parent_education" in df.columns and df["parent_education"].dtype == "object":
        df["parent_education"] = df["parent_education"].map(edu_map).fillna(0).astype(int)

    nominal_cols = [c for c in ["gender", "school_type", "study_method"] if c in df.columns]
    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)

    if scaler is not None and scale_cols:
        cols_to_scale = [c for c in scale_cols if c in df.columns]
        if cols_to_scale:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


def align_columns(df, expected_features):
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    return df[expected_features]


st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Predictor")
st.write("Upload a student CSV file to get **grade predictions**, **classifications**, and **study recommendations**.")

uploaded_file = st.file_uploader("Upload Student Data (CSV)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(raw_df)

    original_df = raw_df.copy()

    # --- FIXED UNPACKING ---
    loaded = load_model()
    if len(loaded) == 4:
        model, feature_names, scaler, scale_cols = loaded
    elif len(loaded) == 2:
        model, feature_names = loaded
        scaler = None
        scale_cols = []
    else:
        raise ValueError("Unexpected number of objects returned from load_model()")

    has_strings = raw_df.select_dtypes(include="object").shape[1] > 0
    if has_strings:
        processed_df = preprocess_raw_data(raw_df, scaler, scale_cols)
    else:
        processed_df = raw_df.copy()
        if "final_grade" in processed_df.columns:
            processed_df.drop(columns=["final_grade"], inplace=True)

    input_df = align_columns(processed_df, feature_names)

    predictions = model.predict(input_df)

    results_df = original_df.copy()
    results_df["Predicted Grade"] = [GRADE_MAP.get(p, f"Grade {p}") for p in predictions]
    results_df["Classification"] = [CATEGORY_MAP.get(p, "Unknown") for p in predictions]

    st.subheader("📊 Predictions & Classifications")
    st.dataframe(results_df)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Grade Distribution**")
        st.bar_chart(results_df["Predicted Grade"].value_counts())
    with col2:
        st.write("**Classification Distribution**")
        st.bar_chart(results_df["Classification"].value_counts())

    st.subheader("💡 Study Recommendations")

    for idx, row in results_df.iterrows():
        student_data = {
            "attendance_percentage": row.get("attendance_percentage", 100),
            "study_hours": row.get("study_hours", 10),
            "math_score": row.get("math_score", 100),
            "science_score": row.get("science_score", 100),
            "english_score": row.get("english_score", 100),
            "internet_access": row.get("internet_access", 1),
        }

        predicted_category = f"Grade {predictions[idx]}"
        recs = generate_recommendations(student_data, predicted_category)

        with st.expander(f"Student {idx + 1} — {row['Predicted Grade']} ({row['Classification']})"):
            for i, r in enumerate(recs, 1):
                st.write(f"{i}. {r}")

    st.subheader("📥 Download Results")
    csv_data = results_df.to_csv(index=False)
    st.download_button("Download Predictions as CSV", csv_data, "predictions.csv", "text/csv")

else:
    st.info("👆 Please upload a CSV file to get started.")