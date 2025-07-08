import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import re
import io

st.set_page_config(page_title="AI 데이터 품질 도구", layout="wide")
st.title("📊 AI 기반 데이터 품질 관리 플랫폼")

uploaded_file = st.file_uploader("📂 CSV 파일 업로드", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # 📘 테이블 설명 입력
    table_description = st.text_area(
        "📝 이 테이블의 데이터에 대한 설명을 입력해주세요:",
        placeholder="예: 이 테이블은 고객의 개인정보와 구매이력을 포함하고 있습니다.")

    # 🔍 기능별 관련 컬럼 분석 요청 (항시 버튼 노출)
    if 'function_column_analysis' not in st.session_state:
        st.session_state['function_column_analysis'] = {}

    if st.button("🧠 각 기능별 관련 컬럼과 이유 분석 요청"):
        if not table_description.strip():
            st.warning("⚠️ 테이블 설명을 먼저 입력해주세요.")
        else:
            st.session_state['function_column_analysis'] = {}
            analysis_tasks = [
                "1) 비정상 데이터 탐지 기능",
                "2) 데이터 결측값 자동 보정 기능",
                "3) 구조(Meta) 결측값 자동 보정 기능",
                "4) 도메인/코드 표준 자동 수정 기능"
            ]
            for task in analysis_tasks:
                prompt = f"""
                            테이블 설명: {table_description}
                            컬럼 목록: {list(df.columns)}
                            기능: {task}
                            해당 기능과 관련성이 높은 컬럼 리스트와 간단한 이유를 알려줘. 너무 자세하지 않게 간결히 설명해줘.
                         """
                response = llm.invoke([HumanMessage(content=prompt)])
                st.session_state['function_column_analysis'][task] = response.content

    if st.session_state.get('function_column_analysis'):
        for task, content in st.session_state['function_column_analysis'].items():
            if f'rendered_{task}' not in st.session_state:
                st.markdown(f"#### 🔍 {task} 관련 컬럼 분석 결과")
                full_text = ""
                placeholder = st.empty()
                for char in content:
                    full_text += char
                    placeholder.markdown(full_text + "▌")
                    import time
                    time.sleep(0.005)
                placeholder.markdown(full_text)
                st.session_state[f'rendered_{task}'] = True
            else:
                st.markdown(f"#### 🔍 {task} 관련 컬럼 분석 결과")
                st.markdown(st.session_state['function_column_analysis'][task])