# app_db.py

import streamlit as st
from src.db_utils import get_db_engine, get_table_names, load_table_data

st.set_page_config(page_title="📡 DB 연결 테스트", layout="wide")
st.title("🧩 PostgreSQL DB 연동 데모")

st.sidebar.header("🛠 DB 접속 정보 입력")
db_user = st.sidebar.text_input("DB 사용자명", "postgres")
db_pass = st.sidebar.text_input("DB 비밀번호", type="password")
db_host = st.sidebar.text_input("DB 호스트", "localhost")
db_port = st.sidebar.text_input("DB 포트", "5432")
db_name = st.sidebar.text_input("DB 이름", "mydatabase")

if st.sidebar.button("🔗 연결 시도"):
    try:
        engine = get_db_engine(db_user, db_pass, db_host, db_port, db_name)
        tables = get_table_names(engine)
        st.success(f"✅ 연결 성공! 테이블 목록: {tables}")

        selected_table = st.selectbox("📋 테이블 선택", tables)
        if selected_table:
            df = load_table_data(engine, selected_table)
            st.dataframe(df)
    except Exception as e:
        st.error(f"❌ 연결 실패: {e}")