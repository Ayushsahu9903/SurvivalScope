import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("titanic_pipeline.joblib")


model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Enter passenger details below to predict the probability of survival, "
    "using a Random Forest trained on the Titanic dataset."
)

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger class", options=[1, 2, 3], index=2,
                           format_func=lambda x: {1: "1st", 2: "2nd", 3: "3rd"}[x])
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.slider("Age", min_value=0, max_value=90, value=30)
    who = st.selectbox("Passenger type", options=["man", "woman", "child"])

with col2:
    fare = st.number_input("Fare paid (£)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    embarked = st.selectbox("Port of embarkation", options=["S", "C", "Q"],
                             format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
    sibsp = st.number_input("Siblings/spouses aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/children aboard", min_value=0, max_value=10, value=0)

# --- Recreate the same engineered features used in training ---
who_to_title = {"man": "Mr", "woman": "Mrs", "child": "Child"}
title = who_to_title.get(who, "Other")
family_size = sibsp + parch + 1

input_df = pd.DataFrame([{
    "pclass": pclass,
    "sex": sex,
    "age": age,
    "fare": fare,
    "embarked": embarked,
    "family_size": family_size,
    "title": title,
}])

st.divider()

if st.button("Predict survival", type="primary", use_container_width=True):
    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success(f"🟢 Predicted: **Survived** — probability {proba:.1%}")
    else:
        st.error(f"🔴 Predicted: **Did not survive** — probability of survival {proba:.1%}")

    st.progress(min(max(proba, 0.0), 1.0))
    with st.expander("Input sent to the model"):
        st.dataframe(input_df)

st.caption("Model: RandomForestClassifier inside a scikit-learn Pipeline (imputation + one-hot encoding + scaling). "
           "Trained on seaborn's built-in Titanic dataset.")
