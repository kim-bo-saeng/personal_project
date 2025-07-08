# 📄 app.py
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

# ✅ 페이지 설정
st.set_page_config(page_title="AI 데이터 품질 도구", layout="wide")
st.title("📊 AI 기반 데이터 품질 관리 플랫폼")

# ✅ API 키 로딩
load_dotenv()

# ✅ 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "question"]
    context: Annotated[str, "context"]
    answer: Annotated[str, "answer"]

# ✅ AI 모델 설정
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ✅ LangGraph 정의
workflow = StateGraph(GraphState)
def handle_question(state: GraphState) -> GraphState:
    return state

def generate_answer(state: GraphState) -> GraphState:
    st.info("💬 AI가 답변을 생성 중입니다...")
    messages = [
        HumanMessage(content=f"질문: {state['question']}\n문맥: {state['context']}")
    ]
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state

workflow.add_node("handle_question", RunnableLambda(handle_question))
workflow.add_node("generate_answer", RunnableLambda(generate_answer))
workflow.set_entry_point("handle_question")
workflow.add_edge("handle_question", "generate_answer")
workflow.add_edge("generate_answer", END)
app = workflow.compile()

# ✅ Streamlit UI
uploaded_file = st.file_uploader("📂 CSV 파일 업로드", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # 📘 테이블 설명 입력
    table_description = st.text_area(
        "📝 이 테이블의 데이터에 대한 설명을 입력해주세요:",
        placeholder="예: 이 테이블은 고객의 개인정보와 구매이력을 포함하고 있습니다."
    )

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

    # 🔍 분석 기능 선택
    with st.container():
        st.markdown("### 🔍 분석 기능 선택")
        selected_analysis = st.selectbox("수행할 분석 기능을 선택하세요:", [
            "1) 비정상 데이터 탐지 기능",
            "2) 데이터 결측값 자동 보정 기능",
            "3) 구조(Meta) 결측값 자동 보정 기능",
            "4) 도메인/코드 표준 자동 수정 기능 (구현 기능)"
        ], key="analysis_selectbox")

    selected_columns = st.multiselect("📌 분석할 컬럼을 선택하세요:", options=df.columns)

    if selected_columns:
        if selected_analysis == "1) 비정상 데이터 탐지 기능":
            st.subheader("🚧 [예정] 비정상 데이터 탐지 기능")
            st.info("향후 구현 예정: 선택된 컬럼의 이상값, 논리적 오류 등을 탐지합니다.")

        elif selected_analysis == "2) 데이터 결측값 자동 보정 기능":
            st.subheader("🚧 [예정] 데이터 결측값 자동 보정 기능")
            st.info("향후 구현 예정: NULL/NaN 값을 AI가 적절하게 채워줍니다.")

        elif selected_analysis == "3) 구조(Meta) 결측값 자동 보정 기능":
            st.subheader("🚧 [예정] 메타데이터 결측값 보정 기능")
            st.info("향후 구현 예정: 데이터 타입, 스키마 정보 누락 부분을 자동 분석합니다.")

        elif selected_analysis == "4) 도메인/코드 표준 자동 수정 기능 (구현 기능)":
            if st.button("🧠 AI 분석 및 정제 수행"):
                refined_df = df.copy()
                st.session_state['refined_df'] = refined_df
                st.session_state['mapping_results'] = {}

                for col in selected_columns:
                    distinct_values = df[col].dropna().unique().tolist()

                    issue_prompt = (
                        f"다음은 '{col}' 컬럼의 DISTINCT 값 리스트야: {distinct_values}\n"
                        "이 컬럼에 데이터 품질 문제가 있을지 판단해줘. "
                        "예: 값이 통일되지 않았거나, 비어 있는 값이 있거나, 잘못된 값이 있는 경우 등 가능한 문제를 모두 설명해줘."
                    )
                    issue_response = llm.invoke([HumanMessage(content=issue_prompt)])

                    refine_prompt = (
                        f"다음은 '{col}' 컬럼의 DISTINCT 값 리스트야: {distinct_values}\n"
                        "동일한 의미인데 표기가 다른 값이 있다면 하나로 통일하고, "
                        "각 원래 값이 어떤 값으로 바뀌어야 하는지 반드시 파이썬 딕셔너리 형태로만 답해줘. "
                        "예: {{'남': '남성', 'M': '남성', '여': '여성'}}"
                    )
                    refine_response = llm.invoke([HumanMessage(content=refine_prompt)])

                    try:
                        mapping = eval(refine_response.content)
                        if isinstance(mapping, dict):
                            refined_df[col] = df[col].map(lambda x: mapping.get(str(x).strip(), x))
                            st.session_state['mapping_results'][col] = {
                                "issue": issue_response.content,
                                "mapping": mapping,
                            }
                        else:
                            st.session_state['mapping_results'][col] = {
                                "issue": issue_response.content,
                                "error": "LLM 응답이 딕셔너리 형식이 아닙니다.",
                            }
                    except Exception as e:
                        st.session_state['mapping_results'][col] = {
                            "issue": issue_response.content,
                            "error": f"매핑 파싱 실패: {e}\n응답: {refine_response.content}",
                        }

            # 결과 출력
            if 'mapping_results' in st.session_state:
                st.subheader("📋 분석 및 정제 결과")

                for col, result in st.session_state['mapping_results'].items():
                    with st.expander(f"📌 [{col}] 컬럼 분석 결과", expanded=False):
                        st.markdown("🔍 **문제점 분석:**")
                        st.write(result.get("issue", "없음"))

                        if "mapping" in result:
                            st.markdown("🔧 **적용된 매핑:**")
                            st.write(result["mapping"])
                            st.markdown("📊 **정제된 데이터 미리보기:**")
                            st.dataframe(st.session_state['refined_df'][[col]].head(10))
                        elif "error" in result:
                            st.error(result["error"])

                csv = st.session_state['refined_df'].to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="💾 정제된 전체 데이터 CSV 다운로드",
                    data=csv,
                    file_name="refined_data.csv",
                    mime="text/csv",
                )
else:
    st.info("📥 CSV 파일을 업로드해주세요.")