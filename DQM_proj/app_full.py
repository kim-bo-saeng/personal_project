# app_full.py

import streamlit as st
import pandas as pd

from src.db_utils import get_db_engine, get_table_names, load_table_data
from src.detect_outliers import detect_outliers_customer
from src.impute_missing import impute_customer_data, impute_order_data
from src.standardize_domain import standardize_gender, standardize_status
from src.infer_meta import infer_customer_meta, infer_order_meta

st.set_page_config(page_title="📊 AI 기반 데이터 품질 관리", layout="wide")
st.title("🔍 DB 기반 AI 데이터 품질 솔루션")

# 📌 DB 접속 정보
st.sidebar.header("📡 DB 접속 정보 입력")
db_user = st.sidebar.text_input("사용자명", "postgres")
db_pass = st.sidebar.text_input("비밀번호", type="password")
db_host = st.sidebar.text_input("호스트", "localhost")
db_port = st.sidebar.text_input("포트", "5432")
db_name = st.sidebar.text_input("DB 이름", "mydatabase")

if st.sidebar.button("🔗 DB 연결"):
    try:
        engine = get_db_engine(db_user, db_pass, db_host, db_port, db_name)
        tables = get_table_names(engine)
        st.session_state.engine = engine
        st.session_state.tables = tables
        st.success("✅ 연결 성공!")
    except Exception as e:
        st.error(f"❌ 연결 실패: {e}")

# ✅ 연결 후 기능 실행
if "engine" in st.session_state:
    table = st.selectbox("📋 분석할 테이블을 선택하세요", st.session_state.tables)

    if table:
        df = load_table_data(st.session_state.engine, table)
        st.write("📄 테이블 미리보기 (상위 100건)")
        st.dataframe(df)

        func = st.selectbox("⚙️ 실행할 기능 선택", [
            "비정상 데이터 탐지",
            "결측값 자동 보정",
            "도메인/코드 표준화",
            "메타 결측 보정"
        ])

        result_df = df.copy()

        if func == "비정상 데이터 탐지":
            if "customer" in table:
                issues = detect_outliers_customer(df)
                for k, v in issues.items():
                    st.markdown(f"**{k.upper()}**")
                    st.dataframe(pd.DataFrame(v))
            else:
                st.warning("❗ 이 기능은 고객 테이블에서만 작동합니다.")

        elif func == "결측값 자동 보정":
            if "customer" in table:
                result_df = impute_customer_data(df)
            else:
                result_df = impute_order_data(df)
            st.success("✅ 보정 완료")
            st.dataframe(result_df)

        elif func == "도메인/코드 표준화":
            if "customer" in table:
                result_df = standardize_gender(df)
            else:
                result_df = standardize_status(df)
            st.success("✅ 표준화 완료")
            st.dataframe(result_df)

        elif func == "메타 결측 보정":
            if "customer" in table:
                missing = infer_customer_meta(df)
            else:
                missing = infer_order_meta(df)
            st.markdown(f"🧩 누락된 메타 컬럼 제안: `{missing}`")

        # 다운로드 버튼
        if func != "메타 결측 보정":
            csv = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ 결과 CSV 다운로드", csv, file_name="result.csv", mime="text/csv")