import streamlit as st
import pandas as pd
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


# --- Load saved model and feature names ---
@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
        features = pickle.load(f)
    return model, features


# --- Align uploaded data columns to match training features ---
def align_columns(df, expected_features):
    """Add missing columns as 0 and reorder to match training data."""
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
    # Read the CSV
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(raw_df, use_container_width=True)

    # Load model
    model, feature_names = load_model()

    # Align columns
    try:
        input_df = align_columns(raw_df.copy(), feature_names)
    except Exception as e:
        st.error(f"Column alignment failed: {e}")
        st.stop()

    # --- Predictions ---
    predictions = model.predict(input_df)

    results_df = raw_df.copy()
    results_df["Predicted Grade"] = [GRADE_MAP.get(p, f"Grade {p}") for p in predictions]
    results_df["Classification"] = [CATEGORY_MAP.get(p, "Unknown") for p in predictions]

    st.subheader("📊 Predictions & Classifications")
    st.dataframe(results_df, use_container_width=True)

    # --- Summary counts ---
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

    # --- Download results ---
    st.subheader("📥 Download Results")
    csv_data = results_df.to_csv(index=False)
    st.download_button("Download Predictions as CSV", csv_data, "predictions.csv", "text/csv")

else:
    st.info("👆 Please upload a CSV file to get started.")
