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

# Load API key
load_dotenv()

# ✅ 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "question"]
    context: Annotated[str, "context"]
    answer: Annotated[str, "answer"]

# ✅ GPT 모델 설정
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ✅ 노드 1: 질문 고정
def handle_question(state: GraphState) -> GraphState:
    state["question"] = (
        "테이블 컬럼의 정보를 기반으로 카테고리형 컬럼이 무엇인지 알려줘. "
        "답변은 컬럼명만 나열해줘."
    )
    return state

# ✅ 노드 2: GPT API 호출
def generate_answer(state: GraphState) -> GraphState:
    st.info("💬 GPT가 답변을 생성 중입니다...")
    messages = [
        HumanMessage(content=f"질문: {state['question']}\n문맥: {state['context']}")
    ]
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state

# ✅ LangGraph 워크플로우 구성
workflow = StateGraph(GraphState)
workflow.add_node("handle_question", RunnableLambda(handle_question))
workflow.add_node("generate_answer", RunnableLambda(generate_answer))
workflow.set_entry_point("handle_question")
workflow.add_edge("handle_question", "generate_answer")
workflow.add_edge("generate_answer", END)
app = workflow.compile()

# ✅ Streamlit UI
st.title("📊 테이블 분석 도우미")
st.write("CSV 파일을 업로드하면 컬럼 정보를 기반으로 카테고리형 컬럼을 탐지하고 선택할 수 있습니다.")

# 1. CSV 업로드
uploaded_file = st.file_uploader("📂 CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("🔎 업로드된 데이터 (미리보기)")
    st.dataframe(df.head())

    # 1. LLM에게 카테고리형 컬럼 추정 요청 (참고용)
    if 'category_analysis_done' not in st.session_state:
        st.session_state['category_analysis_done'] = False
    if 'category_analysis_result' not in st.session_state:
        st.session_state['category_analysis_result'] = None

    show_analysis_ui = False
    if not st.session_state['category_analysis_done']:
        if st.button("GPT에게 카테고리형 컬럼 분석 요청"):
            column_info = f"테이블명: 업로드된 테이블\n컬럼명: {list(df.columns)}"
            input_state = {
                "question": "테이블 컬럼의 정보를 기반으로 카테고리형 컬럼이 무엇인지 알려줘. 답변은 반드시 ['컬럼1', '컬럼2', ...] 형태의 파이썬 리스트로만 해줘. 다른 설명은 하지마.",
                "context": column_info,
                "answer": ""
            }
            result = app.invoke(input_state)
            st.session_state['category_analysis_result'] = result['answer']
            st.session_state['category_analysis_done'] = True
            show_analysis_ui = True
    else:
        show_analysis_ui = True

    if show_analysis_ui:
        st.success("✅ GPT가 추정한 카테고리형 컬럼입니다:")
        if st.session_state['category_analysis_result']:
            st.markdown(f"**🧠 GPT 답변:**\n{st.session_state['category_analysis_result']}")
        st.info("카테고리형 컬럼 분석이 완료되었습니다. 아래에서 컬럼을 선택하고 추가 분석을 진행하세요.")

        selected_columns = st.multiselect(
            "📌 사용할 컬럼을 선택하세요:",
            options=df.columns,
            default=[]
        )

        if selected_columns:
            st.subheader("🎯 선택된 컬럼 데이터 (미리보기)")
            st.dataframe(df[selected_columns].head())

            # 1. 전체 정제 및 다운로드 버튼
            if st.button("선택한 컬럼 전체 정제 및 다운로드", key="refine_all"):
                refined_df = df.copy()
                all_mappings = {}
                mapping_errors = {}
                for col in selected_columns:
                    distinct_values = df[col].dropna().unique().tolist()
                    prompt = (
                        f"다음은 '{col}' 컬럼의 DISTINCT 값 리스트야: {distinct_values}\n"
                        f"동일한 의미인데 표기가 다른 값이 있다면 하나로 통일하고, 각 원래 값이 어떤 값으로 바뀌어야 하는지 반드시 파이썬 딕셔너리 형태로만 답해줘. 예시: {{'남': '남성', '남자': '남성', 'M': '남성', ...}}. 다른 설명은 하지마."
                    )
                    messages = [HumanMessage(content=prompt)]
                    response = llm.invoke(messages)
                    try:
                        mapping = eval(response.content)
                        if isinstance(mapping, dict):
                            all_mappings[col] = mapping
                            refined_df[col] = df[col].map(lambda x: mapping.get(str(x).strip(), x))
                        else:
                            mapping_errors[col] = "답변이 딕셔너리 형식이 아닙니다. 직접 확인 필요."
                    except Exception as e:
                        mapping_errors[col] = f"매핑 파싱 실패: {e}. 답변: {response.content}"

                # 2. 컬럼별 매핑 결과 시각화
                st.markdown("---")
                st.subheader("🧠 컬럼별 정제 매핑 결과")
                for col in selected_columns:
                    with st.expander(f"[{col}] 매핑 결과 보기"):
                        if col in all_mappings:
                            st.write("적용된 매핑:", all_mappings[col])
                        if col in mapping_errors:
                            st.warning(mapping_errors[col])

                # 3. 정제된 전체 DataFrame(원본+정제컬럼) 다운로드
                csv = refined_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label=f"정제된 전체 데이터(원본+정제컬럼) CSV 다운로드",
                    data=csv,
                    file_name=f"refined_full_data.csv",
                    mime='text/csv',
                    key="download_all"
                )
        else:
            st.warning("⚠️ 최소 하나 이상의 컬럼을 선택해주세요.")
else:
    st.info("CSV 파일을 업로드해주세요.")