# app.py

import streamlit as st
import pandas as pd

from src.standardize_domain_code import standardize_code, standardize_status
# from src.detect_outliers import detect_outliers_customer
# from src.impute_missing import impute_customer_data, impute_order_data
# from src.infer_meta import infer_customer_meta, infer_order_meta

st.set_page_config(page_title="AI 데이터 품질 도구", layout="wide")
st.title("📊 AI 기반 데이터 품질 관리 플랫폼")

uploaded_file = st.file_uploader("📁 CSV 파일 업로드", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 업로드한 원본 데이터")
    st.dataframe(df)

    st.subheader("🧹 도메인/코드 정규화")
    df_standard = standardize_domain_code(df)
    st.dataframe(df_standard)
    
    """
    st.subheader("🚨 비정상 데이터 탐지")
    issues = detect_outliers_customer(df)
    for key, records in issues.items():
        st.markdown(f"**{key.upper()}**")
        st.dataframe(pd.DataFrame(records))

    st.subheader("🛠 결측값 자동 보정")
    df_imputed = impute_customer_data(df)
    st.dataframe(df_imputed)

    st.subheader("🧩 메타 결측 보정")
    missing_meta = infer_customer_meta(df)
    st.markdown(f"필요한 메타 컬럼 제안: `{missing_meta}`")

    """
    st.subheader("⬇️ 보정/표준화된 데이터 다운로드")
    csv = df_standard.to_csv(index=False).encode('utf-8-sig')
    st.download_button("CSV 다운로드", csv, file_name="cleaned_data.csv", mime="text/csv")
    