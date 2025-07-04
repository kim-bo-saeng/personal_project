# 📄 app.py
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load API key
load_dotenv()

# ✅ 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "question"]
    context: Annotated[str, "context"]
    answer: Annotated[str, "answer"]

# ✅ GPT 모델 설정
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ✅ 노드 1: 질문 + context 생성
def handle_question(state: GraphState) -> GraphState:
    # 질문은 고정
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
st.write("테이블 컬럼 정보(context)를 입력하면, 카테고리형 컬럼을 분석해 드립니다.")

# 사용자 입력 받기
user_context = st.text_area("📥 테이블 정보 (context)를 입력하세요", height=200, placeholder="예: 테이블명 : cust_info, 컬럼명 : [고객아이디,이름,나이,...]")

if st.button("답변 생성"):
    if not user_context.strip():
        st.warning("⚠️ 테이블 정보(context)를 입력해주세요.")
    else:
        # 초기 상태
        input_state = {
            "question": "",       # 질문은 handle_question에서 고정
            "context": user_context,
            "answer": ""
        }

        # 그래프 실행
        result = app.invoke(input_state)

        # 결과 출력
        st.success("✅ 답변이 생성되었습니다:")
        st.markdown(f"**🧠 GPT 답변:**\n{result['answer']}")