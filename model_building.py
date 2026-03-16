import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ================= Page Config =================
st.set_page_config(
    page_title="Travel Package Prediction",
    page_icon="✈️",
    layout="centered"
)

# ================= Header =================
st.title("✈️ Travel Package Purchase Prediction")
st.write("Predict whether a customer is likely to purchase a travel package")

st.divider()


# ================= Load Dataset =================
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Hp\Desktop\MACHINE LEARNING\ML project final.csv")
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")

    if "CustomerID" in df.columns:
        df.drop("CustomerID", axis=1, inplace=True)

    if "AgeGroup" in df.columns:
        df.drop("AgeGroup", axis=1, inplace=True)

    return df


df = load_data()


# ================= Features & Target =================
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns


# ================= Train-Test Split =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ================= Preprocessing =================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])


# ================= Model Pipeline =================
@st.cache_resource
def train_model(X_train, y_train):
    model_pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model_pipeline.fit(X_train, y_train)
    return model_pipeline


model_pipeline = train_model(X_train, y_train)


# ================= Model Evaluation =================
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)


# ================= Show Metrics =================
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("✅ Accuracy", f"{accuracy:.2f}")

with col2:
    st.metric("🎯 Precision", f"{precision:.2f}")

st.divider()


# ================= Input Section =================
st.subheader("📝 Customer Details")

user_input = {}

# Numeric inputs
for i in range(0, len(num_cols), 3):
    cols = st.columns(3)
    for idx, col in enumerate(num_cols[i:i+3]):
        user_input[col] = cols[idx].number_input(
            col,
            value=float(X[col].median())
        )

# Categorical inputs
for i in range(0, len(cat_cols), 3):
    cols = st.columns(3)
    for idx, col in enumerate(cat_cols[i:i+3]):
        user_input[col] = cols[idx].selectbox(
            col,
            options=X[col].unique().tolist()
        )

st.divider()


# ================= Prediction =================
if st.button("🔍 Predict Outcome", use_container_width=True):
    input_df = pd.DataFrame([user_input])
    proba = model_pipeline.predict_proba(input_df)[0][1]

    st.subheader("📈 Prediction Result")
    st.metric("Purchase Probability", f"{proba:.2f}")

    if proba >= 0.5:
        st.success("✅ Customer is likely to purchase")
    else:
        st.error("❌ Customer is unlikely to purchase")
