import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ================= Streamlit Page Config =================
st.set_page_config(layout="centered")
st.title("📊 Exploratory Data Analysis (EDA)")
st.title("✈️ Comprehensive Analysis of Holiday Product Uptake")
st.write("This app shows the EDA analysis of the ML project dataset.")

# ================= Load Dataset =================
df = pd.read_csv(r"C:\Users\Hp\Desktop\MACHINE LEARNING\ML project final.csv")

# Identify numeric and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# ================= Dataset Overview =================
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

st.subheader("📐 Dataset Shape")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

import io
buffer = io.StringIO()
df.info(buf=buffer)
#st.text(buffer.getvalue())

st.subheader("❗ Missing Values")
st.dataframe(df.isnull().sum())

st.subheader("📈 Statistical Summary")
st.dataframe(df.describe())

# ================= Visualizations =================
st.subheader("🔹 Univariate Analysis")
st.write("Distribution and frequency of individual features")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numeric Feature Distribution**")
        uni_num = st.selectbox("Select numeric column", num_cols, key="uni_num")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(x=df[uni_num], ax=ax)
        st.pyplot(fig)
    with col2:
        st.markdown("**Categorical Feature Count**")
        uni_cat = st.selectbox("Select categorical column", cat_cols, key="uni_cat")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.countplot(y=df[uni_cat], ax=ax)
        st.pyplot(fig)

# ================= Bivariate Analysis =================
st.subheader("🔹 Bivariate Analysis")
st.write("Relationship between two variables.")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numeric vs Numeric**")
        x_col = st.selectbox("Select X-axis", num_cols, key="bi_x")
        y_col = st.selectbox("Select Y-axis", num_cols, key="bi_y")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
        st.pyplot(fig)
    with col2:
        st.markdown("**Categorical vs Numeric**")
        cat_col = st.selectbox("Select categorical column", cat_cols, key="bi_cat")
        num_col = st.selectbox("Select numeric column", num_cols, key="bi_num")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
        st.pyplot(fig)

