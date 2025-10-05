# import streamlit as st
# import pandas as pd
# import joblib

# # Load trained model
# model = joblib.load("model.pkl")

# st.title("ScreenTime vs Mental Wellness Predictor (Local)")

# st.write("Enter the feature values to predict mental wellness index (0-100)")

# # Load CSV to detect feature names
# df = pd.read_csv("ScreenTime vs MentalWellness1.csv")
# features = df.drop(columns=["mental_wellness_index_0_100"]).columns

# # Collect input values dynamically
# input_data = {}
# for f in features:
#     # If column is numeric
#     if pd.api.types.is_numeric_dtype(df[f]):
#         input_data[f] = st.number_input(f"{f}:", value=float(df[f].mean()))
#     # If column is categorical
#     else:
#         options = df[f].unique().tolist()
#         input_data[f] = st.selectbox(f"{f}:", options)

# # Predict button
# if st.button("Predict"):
#     X_input = pd.DataFrame([input_data])

#     # One-hot encode categorical columns
#     X_input = pd.get_dummies(X_input)

#     # Ensure all training columns exist
#     for col in model.feature_names_in_:
#         if col not in X_input.columns:
#             X_input[col] = 0

#     # Reorder columns to match training
#     X_input = X_input[model.feature_names_in_]

#     # Prediction
#     prediction = model.predict(X_input)[0]
#     st.success(f"Predicted Mental Wellness Index: {prediction:.2f}")

# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt

# # Load trained model
# model = joblib.load("model.pkl")

# # Page config
# st.set_page_config(page_title="ScreenTime vs Mental Wellness", layout="wide")

# st.title("🧠 ScreenTime vs Mental Wellness Predictor")
# st.markdown(
#     "Enter your daily habits and lifestyle data below to predict your Mental Wellness Index (0-100)."
# )

# # Load CSV to detect features
# df = pd.read_csv("ScreenTime vs MentalWellness1.csv")
# features = df.drop(columns=["mental_wellness_index_0_100"]).columns

# # Columns layout
# col1, col2 = st.columns(2)
# input_data = {}

# # Alternate features between columns
# for i, f in enumerate(features):
#     target_col = col1 if i % 2 == 0 else col2  # even index -> col1, odd index -> col2
#     if pd.api.types.is_numeric_dtype(df[f]):
#         min_val, max_val, mean_val = float(df[f].min()), float(df[f].max()), float(df[f].mean())
#         input_data[f] = target_col.slider(f"{f}:", min_val, max_val, mean_val)
#     else:
#         options = df[f].unique().tolist()
#         input_data[f] = target_col.selectbox(f"{f}:", options)

# # Predict button
# if st.button("Predict Mental Wellness Index"):
#     X_input = pd.DataFrame([input_data])

#     # One-hot encode categorical columns
#     X_input = pd.get_dummies(X_input)

#     # Ensure all training columns exist
#     for col in model.feature_names_in_:
#         if col not in X_input.columns:
#             X_input[col] = 0

#     # Reorder columns to match training
#     X_input = X_input[model.feature_names_in_]

#     # Prediction
#     prediction = model.predict(X_input)[0]

#     st.markdown("---")
#     st.subheader("🧮 Prediction Result")
#     st.success(f"**Predicted Mental Wellness Index:** {prediction:.2f} / 100")
#     st.markdown(
#         "💡 This prediction is based on your input and a locally trained model. "
#         "For informational purposes only."
#     )

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load trained model
model = joblib.load("model.pkl")

# Page config
st.set_page_config(page_title="Mental Wellness Prediction", layout="centered")
st.title("🧠 Mental Wellness Index Prediction using ML Model")
st.write("Provide details to estimate your **Mental Wellness Index (0-100)**")
st.markdown("---")

# ----------------------------
# Input UI
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=80, value=25, step=1)  # step = 1 (fix)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary/Other"])
    occupation = st.selectbox("Occupation", ["Student", "Employed", "Unemployed", "Self-employed", "Retired"])
    work_mode = st.selectbox("Work Mode", ["Remote", "In-person", "Hybrid"])

with col2:
    work_screen_hours = st.number_input("Work Screen Hours", min_value=0.0, max_value=24.0, value=4.0, step=0.5)
    leisure_screen_hours = st.number_input("Leisure Screen Hours", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
    total_screen_time = work_screen_hours + leisure_screen_hours
    st.text_input("Total Screen Time (hours/day)", value=str(total_screen_time), disabled=True)  # auto-calc
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.25)

# Sliders (customization 1)
sleep_quality = st.slider("Sleep Quality (1-5)", min_value=1, max_value=5, value=3)
stress_level = st.slider("Stress Level (0-10)", min_value=0, max_value=10, value=5)
productivity = st.slider("Productivity (0-100)", min_value=0, max_value=100, value=50)

# More inputs
exercise_minutes = st.number_input("Exercise Minutes per Week", min_value=0, max_value=1000, value=120, step=10)
social_hours = st.number_input("Social Hours per Week", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("🔮 Predict Mental Wellness Index", use_container_width=True):
    input_data = [[
        age, gender, occupation, work_mode,
        total_screen_time, work_screen_hours, leisure_screen_hours,
        sleep_hours, sleep_quality, stress_level, productivity,
        exercise_minutes, social_hours
    ]]

    # Convert to DataFrame
    X_input = pd.DataFrame(input_data, columns=[
        "age", "gender", "occupation", "work_mode",
        "screen_time_hours", "work_screen_hours", "leisure_screen_hours",
        "sleep_hours", "sleep_quality_1_5", "stress_level_0_10", "productivity_0_100",
        "exercise_minutes_per_week", "social_hours_per_week"
    ])

    # One-hot encode categorical columns
    X_input = pd.get_dummies(X_input)

    # Ensure all training columns exist
    for col in model.feature_names_in_:
        if col not in X_input.columns:
            X_input[col] = 0

    # Match training column order
    X_input = X_input[model.feature_names_in_]

    # Prediction
    prediction = model.predict(X_input)[0]

    def get_status(score):
        if score <= 20:
            return "🚨 Not Fit (Severe imbalance, needs urgent attention)", "red"
        elif score <= 40:
            return "⚠️ Vulnerable (Struggling, high stress & low resilience)", "orange"
        elif score <= 60:
            return "😐 Moderate (Average wellness, could improve with lifestyle balance)", "yellow"
        elif score <= 80:
            return "🙂 Healthy (Good balance, manageable stress, positive habits)", "lightgreen"
        else:
            return "🌟 Mentally Fit (Strong resilience, high wellness, thriving)", "green"
            
    def get_suggestions(status):
        if status == "🚨 Not Fit (Severe imbalance, needs urgent attention)":
            return [
                "🧑‍⚕️ Seek professional mental health support.",
                "🛌 Reduce screen time and prioritize 7–8 hours of sleep.",
                "📝 Start with small habits: walking, journaling, meditation.",
                "🤝 Stay connected with supportive friends/family."
            ], "#ffe6e6"  # light red background
        elif status == "⚠️ Vulnerable (Struggling, high stress & low resilience)":
            return [
                "🛌 Maintain a fixed sleep routine.",
                "📴 Limit unnecessary screen use (e.g., social media).",
                "🧘 Practice stress relief: yoga, breathing, mindfulness.",
                "🏃 Exercise at least 20–30 minutes daily."
            ], "#fff5e6"  # light orange background
        elif status == "😐 Moderate (Average wellness, could improve with lifestyle balance)":
            return [
                "⚖️ Balance work vs. leisure screen hours.",
                "🎭 Increase social interaction and hobbies.",
                "📊 Track sleep, exercise, and diet consistency.",
                "🎯 Set small weekly improvement goals."
            ], "#ffffe6"  # light yellow background
        elif status == "🙂 Healthy (Good balance, manageable stress, positive habits)":
            return [
                "🥗 Keep exercising and eating balanced meals.",
                "🧘 Continue mindfulness/stress management routines.",
                "⏰ Stick to a consistent sleep schedule.",
                "🚀 Avoid overworking and burnout."
            ], "#e6ffe6"  # light green background
        else:  # 🌟 Mentally Fit
            return [
                "🌱 Maintain healthy habits — don't get complacent.",
                "🤝 Support others with wellness advice.",
                "📖 Try advanced practices: gratitude journaling, digital detox.",
                "🧘 Explore mindfulness retreats or volunteering."
            ], "#e6f7ff"  # light blue background


    # NEW: add status + gauge
    status, color = get_status(prediction)

    st.success(f"🎯 Your Predicted Mental Wellness Index: **{prediction:.2f} / 100**")
    st.markdown(f"**Wellness Status:** <span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)

    suggestions, bg_color = get_suggestions(status)

    st.markdown("---")
    st.subheader("💡 Personalized Suggestions")

    # Card Style Box with Dynamic Color
    st.markdown(
        f"""
        <div style='background-color:{bg_color};
                    padding:15px;
                    border-radius:12px;
                    box-shadow:0 4px 6px rgba(0,0,0,0.1);'>
            <h4 style='color:#111;'>Status: {status}</h4>
            <ul style='color:#222; font-size:16px;'>
            <ul>
                {''.join([f"<li style='margin:8px 0; font-size:16px;'>{s}</li>" for s in suggestions])}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        title = {'text': "Mental Wellness Index"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 20], 'color': "red"},
                {'range': [20, 40], 'color': "orange"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "green"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.8,
                'value': prediction
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)



    # # --- Mini Visualization ---
    # st.subheader("📊 Lifestyle Overview")

    # # Bar chart for screen time distribution
    # screen_features = ["screen_time_hours", "work_screen_hours", "leisure_screen_hours"]
    # screen_values = [input_data[f] for f in screen_features]
    # fig1, ax1 = plt.subplots()
    # ax1.bar(screen_features, screen_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    # ax1.set_ylabel("Hours")
    # ax1.set_title("Screen Time Distribution")
    # st.pyplot(fig1)

    # # Scatter plot: sleep hours vs predicted index
    # fig2, ax2 = plt.subplots()
    # ax2.scatter(input_data["sleep_hours"], prediction, color='purple', s=100)
    # ax2.set_xlim(0, 12)
    # ax2.set_ylim(0, 100)
    # ax2.set_xlabel("Sleep Hours")
    # ax2.set_ylabel("Predicted Mental Wellness Index")
    # ax2.set_title("Sleep Hours vs Predicted Wellness Index")
    # st.pyplot(fig2)
    # st.markdown(
    #     "📈 These charts provide a quick overview of your lifestyle factors related to mental wellness."
    # )   
