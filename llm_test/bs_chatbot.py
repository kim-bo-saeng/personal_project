import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

st.set_page_config(page_title="PDF QA 챗봇", page_icon="📚", layout="wide")
st.title("📚 PDF 기반 대화형 챗봇")
st.write("PDF를 업로드하고, 문서 내용에 대해 대화하듯 질문하세요.")

# 세션 상태
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",   # ★ 중요
        output_key="answer"     # ★ 중요
    )
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# PDF 업로드
uploaded_file = st.sidebar.file_uploader("📂 PDF 업로드", type=["pdf"])
if uploaded_file:
    st.sidebar.success(f"업로드됨: {uploaded_file.name}")

    # 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # 로드/분할
    docs = PyPDFLoader(tmp_path).load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

    # 임베딩/벡터스토어
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)

    # 대화형 RAG 체인 (메모리 포함)
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=st.session_state.memory,          # ← 메모리 주입
        return_source_documents=True
    )

# 기존 대화 렌더링
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.qa_chain is None:
            answer = "먼저 PDF 파일을 업로드해주세요 📂"
        else:
            result = st.session_state.qa_chain({"question": prompt})  # ← 'question' 키 사용
            answer = result["answer"]
            # 참고: result["source_documents"] 에서 출처 열람 가능
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# 초기화
if st.sidebar.button("🗑️ 대화 초기화"):
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    st.session_state.qa_chain = None
    st.rerun()