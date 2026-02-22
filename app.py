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
GRADE_MAP = {0: "Grade F", 1: "Grade E", 2: "Grade D", 3: "Grade C", 4: "Grade B", 5: "Grade A"}
CATEGORY_MAP = {0: "At-Risk", 1: "At-Risk", 2: "Average", 3: "Average", 4: "High-Performing", 5: "High-Performing"}


@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
        features = pickle.load(f)
    return model, features


def preprocess_raw_data(df):
    """
    Preprocess raw student CSV to match the training format.
    Handles: drop student_id, encode categoricals, one-hot encode nominals.
    """
    df = df.copy()

    # Drop student_id if present
    if "student_id" in df.columns:
        df.drop(columns=["student_id"], inplace=True)

    # Drop final_grade if present (we are predicting it)
    if "final_grade" in df.columns:
        df.drop(columns=["final_grade"], inplace=True)

    # Lowercase all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Binary encode: yes/no → 1/0
    binary_map = {"yes": 1, "no": 0}
    for col in ["internet_access", "extra_activities"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    # Ordinal encode: travel_time
    travel_map = {"<15 min": 0, "15-30 min": 1, "30-60 min": 2, ">60 min": 3}
    if "travel_time" in df.columns and df["travel_time"].dtype == "object":
        df["travel_time"] = df["travel_time"].map(travel_map).fillna(0).astype(int)

    # Ordinal encode: parent_education
    edu_map = {"no formal": 0, "high school": 1, "diploma": 2, "graduate": 3, "post graduate": 4, "phd": 5}
    if "parent_education" in df.columns and df["parent_education"].dtype == "object":
        df["parent_education"] = df["parent_education"].map(edu_map).fillna(0).astype(int)

    # One-hot encode: gender, school_type, study_method
    nominal_cols = [c for c in ["gender", "school_type", "study_method"] if c in df.columns]
    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)

    return df


def align_columns(df, expected_features):
    """Add missing columns as 0, remove extra columns, reorder."""
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    return df[expected_features]


# --- Page Config ---
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Predictor")
st.write("Upload a student CSV file to get **grade predictions**, **classifications**, and **study recommendations**.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload Student Data (CSV)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(raw_df)

    # Save original scores before preprocessing (for recommendations)
    original_df = raw_df.copy()

    # Load model and feature names
    model, feature_names = load_model()

    # Check if data needs preprocessing (has string columns)
    has_strings = raw_df.select_dtypes(include="object").shape[1] > 0
    if has_strings:
        processed_df = preprocess_raw_data(raw_df)
    else:
        processed_df = raw_df.copy()
        if "final_grade" in processed_df.columns:
            processed_df.drop(columns=["final_grade"], inplace=True)

    # Align to expected features
    input_df = align_columns(processed_df, feature_names)

    # --- Predictions ---
    predictions = model.predict(input_df)

    results_df = original_df.copy()
    results_df["Predicted Grade"] = [GRADE_MAP.get(p, f"Grade {p}") for p in predictions]
    results_df["Classification"] = [CATEGORY_MAP.get(p, "Unknown") for p in predictions]

    st.subheader("📊 Predictions & Classifications")
    st.dataframe(results_df)

    # --- Summary charts ---
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Grade Distribution**")
        st.bar_chart(results_df["Predicted Grade"].value_counts())
    with col2:
        st.write("**Classification Distribution**")
        st.bar_chart(results_df["Classification"].value_counts())

    # --- Recommendations ---
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

    # --- Download ---
    st.subheader("📥 Download Results")
    csv_data = results_df.to_csv(index=False)
    st.download_button("Download Predictions as CSV", csv_data, "predictions.csv", "text/csv")

else:
    st.info("👆 Please upload a CSV file to get started.")
