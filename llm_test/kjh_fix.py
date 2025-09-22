import os
import tempfile
from dotenv import load_dotenv

# LangChain - OpenAI 모델
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 기타
from langdetect import detect
import streamlit as st


# ==========================
# 🔹 환경변수 및 모델 초기화
# ==========================
load_dotenv()
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-3.5-turbo"
)


# ==========================
# 📌 문서 로드 및 RAG 체인 구성
# ==========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def build_chain_from_files(uploaded_files):
    documents = []
    for file in uploaded_files:
        # 업로드 파일을 임시 파일로 저장
        suffix = "." + file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        # 파일 확장자 확인 후 로드
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            continue
        documents.extend(loader.load())

    if not documents:
        return None, 0

    splits = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)

    chain_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return chain_qa, len(splits)


def query(chain_qa, question):
    result = chain_qa({"query": question})
    return result["result"]


# ==========================
# 📌 외국어 입력 처리
# ==========================
def translate_to_korean(text: str) -> str:
    return text  # TODO: 실제 번역 API 연결

def translate_from_korean(text: str, target_lang: str) -> str:
    return text  # TODO: 실제 번역 API 연결

def foreign_chatbot(message):
    detected_language = detect(message)
    if detected_language != "ko":
        translated_message = translate_to_korean(message)
        response = st.session_state.conversation.predict(input=translated_message)
        return translate_from_korean(response, detected_language)
    return st.session_state.conversation.predict(input=message)


# ==========================
# 📌 Streamlit UI
# ==========================
st.set_page_config(
    page_title="챗봇",
    page_icon="💬",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
    st.session_state.conversation = ConversationChain(
        memory=st.session_state.memory,
        llm=llm
    )

if "chain_qa" not in st.session_state:
    st.session_state.chain_qa = None

st.title("📚 문서 기반 챗봇")
st.write("PDF 또는 TXT 파일을 업로드하고, 문서 내용에 대해 질문하세요!")

# 🔹 파일 업로드 UI
uploaded_files = st.sidebar.file_uploader(
    "파일 업로드 (PDF/TXT 지원)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)}개 파일 업로드 완료 ✅")
    chain_qa, n_chunks = build_chain_from_files(uploaded_files)
    if chain_qa:
        st.session_state.chain_qa = chain_qa
        st.sidebar.write(f"문서가 {n_chunks} 개의 청크로 분할되었습니다.")
    else:
        st.sidebar.error("지원되지 않는 파일 형식입니다.")

# 🔹 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 🔹 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.chain_qa:
            answer = query(st.session_state.chain_qa, prompt)
        else:
            answer = st.session_state.conversation.predict(input=prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# 🔹 초기화 버튼
if st.sidebar.button("대화 기록 초기화"):
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory()
    st.session_state.conversation = ConversationChain(
        memory=st.session_state.memory,
        llm=llm
    )
    st.session_state.chain_qa = None
    st.rerun()