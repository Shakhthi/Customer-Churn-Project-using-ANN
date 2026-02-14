import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle 

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    h1, h2, h3 {color: #2c3e50;}
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
    }
    .stFileUploader {
        border: 2px dashed #2c3e50;
        border-radius: 8px;
        padding: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "EDA", "Prediction", "About"])

# -------------------------------
# Upload Data Page
# -------------------------------
if page == "Upload Data":
    st.header("📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data uploaded successfully!")
        st.dataframe(df.head())
    else:
        st.info("Please upload a dataset to continue.")

# -------------------------------
# EDA Page
# -------------------------------
elif page == "EDA":
    st.header("🔍 Exploratory Data Analysis")
    if "df" in st.session_state:
        df = st.session_state.df
        st.subheader("Dataset Overview")
        st.write(df.describe())
        st.bar_chart(df["Exited"].value_counts())
    else:
        st.warning("Upload a dataset first.")

# -------------------------------
# Prediction Page
# -------------------------------
elif page == "Prediction":
    st.header("🤖 Customer Churn Prediction")
    if "df" in st.session_state:
        model_file = "artifacts/ann_churn_model.keras"

        if model_file:
            model = tf.keras.models.load_model(model_file)
            st.success("✅ Model loaded successfully!")

            st.subheader("Make a Prediction")
            # Example: user inputs
            creditscore = st.slider("Credit Score", 300,450,990)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 30, 110)
            tenure = st.slider("Tenure (years)", 0, 3, 20)
            balance = st.slider("Account Balance", 0, 250000, 1000000)
            numofproducts = st.slider("Number of Products", 1, 3, 6)
            hascrcard = st.selectbox("Has Credit Card?", ["Yes", "No"])
            isactivemember = st.selectbox("Is Active Member?", ["Yes", "No"])
            estimatedsalary = st.slider("Estimated Salary", 0, 200000, 1000000)

            if st.button("Predict"):
                # Dummy preprocessing (replace with your pipeline)
                input_data = pd.DataFrame({
                    "CreditScore": [creditscore],
                    "Geography": [geography],
                    "Gender": [gender],
                    "Age": [age],
                    "Tenure": [tenure],
                    "Balance": [balance],
                    "NumOfProducts": [numofproducts],
                    "HasCrCard": [1 if hascrcard == "Yes" else 0],
                    "IsActiveMember": [1 if isactivemember == "Yes" else 0],
                    "EstimatedSalary": [estimatedsalary]
                }) 
                input_df = pd.DataFrame(input_data)

                with open("artifacts/column_transformer.pkl", "rb") as file:
                    ct = pickle.load(file)


                input_transformed = ct.transform(input_df)
                prediction = model.predict(input_transformed)
                churn_prob = prediction[0][0]

                st.metric("Churn Probability", f"{churn_prob:.2%}")
                if churn_prob > 0.5:
                    st.error("⚠️ High risk of churn")
                else:
                    st.success("✅ Low risk of churn")
        else:
            st.info("Please upload your trained model to continue.")
    else:
        st.warning("Upload a dataset first.")

# -------------------------------
# About Page
# -------------------------------
elif page == "About":
    st.header("ℹ️ About This App")
    st.write("""
        This app demonstrates a **Customer Churn Prediction** workflow:
        - Upload customer dataset
        - Perform quick EDA
        - Load trained ANN model
        - Predict churn probability interactively

        Designed with a modern, recruiter‑friendly UI ✨
    """)
