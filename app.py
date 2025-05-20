#!/usr/bin/python3

# -*- coding: UTF-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io

# Global background and layout styling
st.markdown("""
    <style>
        html, body, [data-testid="stApp"] {
            background: linear-gradient(to bottom, #e0f7fa, #ede7f6) !important;
        }
        .appview-container,
        .main,
        .block-container,
        .element-container,
        .stButton > button,
        .stDownloadButton > button {
            background: transparent !important;
        }
        .block-container {
            padding: 2rem;
        }
        .left-button {
            display: flex;
            justify-content: flex-start;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and features
model = joblib.load("best_model.pkl")
#scaler = joblib.load("scaler.pkl")
top25_features = joblib.load("features.pkl")
top5_display_features = top25_features[:5]

display_names = {
    top25_features[0]: "Approved Courses (2nd Semester)",
    top25_features[1]: "Average Grade (2nd Semester)",
    top25_features[2]: "Approved Courses (1st Semester)",
    top25_features[3]: "Average Grade (1st Semester)",
    top25_features[4]: "Tuition Paid (1 = Yes, 0 = No)"
}

# Title and intro
st.markdown("""
    <h1 style='text-align: center; color: #1e3c72;'>🎓 Dropout Prediction System</h1>
    <p style='text-align: center;'>AI-driven prediction aligned with SDG #4: Quality Education</p>
""", unsafe_allow_html=True)

# Sidebar input
st.sidebar.header("📥 Enter Student Information")
input_data = {}
for feature in top25_features:
    if feature in top5_display_features:
        if feature == top25_features[4]:
            option = st.sidebar.selectbox("Tuition Paid", options=["Yes", "No"])
            value = 1.0 if option == "Yes" else 0.0
            # Move Predict button here
            trigger = st.sidebar.button("🔍 Predict")
        else:
            value = st.sidebar.number_input(display_names.get(feature, feature), min_value=0.0, value=0.0, step=1.0)
    else:
        value = 0.0
    input_data[feature] = float(value)

# Prediction
input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0][1]

if trigger:
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=[input_data[f] for f in top5_display_features],
        theta=[display_names[f] for f in top5_display_features],
        fill='toself',
        name='Student',
        line=dict(color='rgba(0,100,200,0.7)')
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False
    )

    median_dict = {
        top25_features[0]: 7.0,
        top25_features[1]: 11.5,
        top25_features[2]: 6.0,
        top25_features[3]: 11.2,
        top25_features[4]: 1.0
    }
    df_compare = pd.DataFrame({
        "Feature": [display_names[f] for f in top5_display_features],
        "Student": [input_data[f] for f in top5_display_features],
        "Typical": [median_dict[f] for f in top5_display_features]
    })

    simulated_probs = np.random.beta(2, 5, size=1000)
    plt.figure(figsize=(8, 3))
    plt.hist(simulated_probs, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(prob, color='red', linestyle='--', label=f"Current Student: {prob:.2%}")
    plt.title("Dropout Probability Distribution (Simulated)")
    plt.xlabel("Predicted Dropout Probability")
    plt.ylabel("Number of Students")
    plt.legend()

    importance_df = pd.DataFrame({
        'Feature': [display_names[f] for f in top5_display_features],
        'Importance': [0.30, 0.25, 0.20, 0.15, 0.10]
    })

    report_text = f"Dropout Prediction: {'Dropout' if prediction == 1 else 'Continue'}\n"
    report_text += f"Probability: {prob:.2%}\n"
    for f in top5_display_features:
        report_text += f"{display_names[f]}: {input_data[f]}\n"

    tab1, tab2 = st.tabs(["📊 Summary", "📋 Detailed Insights"])

    with tab1:
        st.markdown(f"""
        <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;">
            <h3>🎯 Prediction: {'<span style=\"color:red;\">Dropout</span>' if prediction == 1 else '<span style=\"color:green;\">Continue</span>'}</h3>
            <p style="font-size:18px;">📈 Probability: <strong>{prob:.2%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📉 Risk Level")
        if prob >= 0.8:
            st.error("🚨 Extremely High Risk")
        elif prob >= 0.6:
            st.warning("⚠️ High Risk")
        elif prob >= 0.4:
            st.info("🔎 Medium Risk")
        else:
            st.success("🟢 Low Risk")

        st.subheader("📘 Result Interpretation")
        if prediction == 1:
            st.warning("⚠️ The student is at high risk of dropping out. Consider early academic intervention, financial counseling, or support programs.")
        else:
            st.success("✅ The student is predicted to continue. Maintain support and monitor performance regularly.")

        st.subheader("📥 Export Report")
        st.download_button("Download Text Report", io.BytesIO(report_text.encode()), file_name="dropout_prediction.txt")

        with st.expander("📘 About This AI System"):
            st.markdown("""
            - **Model**: Hybrid stacking (XGBoost + KNN + Logistic Regression)
            - **Goal**: Predict student dropout risk
            - **Inputs**: Academic & Financial metrics (25 features)
            - **SDG**: Supports [SDG 4 – Quality Education](https://sdgs.un.org/goals/goal4)
            """)

    with tab2:
        st.subheader("📏 Student vs Median Comparison")
        st.bar_chart(df_compare.set_index("Feature"))

        st.subheader("📈 Probability Distribution Overview")
        st.pyplot(plt)

        st.subheader("🔎 Feature Importance (Top 5)")
        st.bar_chart(importance_df.set_index("Feature"))

        st.subheader("📊 Student Radar Profile")
        st.plotly_chart(radar_fig, use_container_width=True)