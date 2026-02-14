import streamlit as st
import pandas as pd

st.title("Spending Analyzer (Minimal)")
st.write(
    "Upload a CSV file with at least two columns: `Category` and `Amount`.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    if {"Category", "Amount"}.issubset(df.columns):
        summary = df.groupby("Category")["Amount"].sum()
        st.bar_chart(summary)
    else:
        st.error("CSV must contain 'Category' and 'Amount' columns.")
