import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# 환경 변수 불러오기 부분
# -----------------------------
load_dotenv()
# 원본에서는 os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 를 바로 넣었는데,
# dotenv를 통해 환경 변수를 불러온 뒤 os.environ에 세팅하는 게 더 안전합니다.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -----------------------------
# PDF 폴더 경로 설정
# -----------------------------
# 원본 코드에서는 "./C:\Users\..." 형태로 잘못 작성되어 escape 문제가 생김.
# → Path 객체 + raw string(r"") 사용으로 수정.
folder = Path(r"C:\Users\user\Documents\GitHub\personal_project\llm_test")

# -----------------------------
# 텍스트 저장 리스트 초기화
# -----------------------------
text = []

# -----------------------------
# 텍스트 분할기 정의
# -----------------------------
# 원본과 동일하지만 유지 이유 설명 추가.
text_splitter = CharacterTextSplitter(
    chunk_size=1000,   # 문서를 1000자 단위로 분할
    chunk_overlap=50,  # 50자 겹치게 분할
    separator="\n"     # 줄바꿈 기준으로 분할
)

# -----------------------------
# PDF 파일 로드 및 텍스트 분할
# -----------------------------
# 원본에서는 문자열 연결 (folder + '/' + file) 사용 → 경로 문제 발생 가능.
# Path 객체를 사용하여 더 안전하게 수정.
for file in os.listdir(folder):
    if file.endswith(".pdf"):
        raw_paper = PyPDFLoader(folder / file).load()
        paper = text_splitter.split_documents(raw_paper)
        text.extend(paper)

# -----------------------------
# 벡터 DB 생성
# -----------------------------
database = Chroma.from_documents(text, OpenAIEmbeddings())
retriever = database.as_retriever(search_kwargs={"k": 5})  # 검색 결과 상위 5개 반환

# -----------------------------
# 프롬프트 템플릿 정의
# -----------------------------
# 원본에서는 "\ns\n" 같은 불필요한 escape 문자열이 들어있었음.
# → 가독성을 위해 "\n\n" 으로 정리.
prompt = ChatPromptTemplate.from_template(
    "당신은 질문-답변(Question-Answer) Task를 수행하는 AI 어시스턴트입니다."
    " 검색된 문맥(context)을 사용하여 질문(question)에 답하세요."
    " 만약 문맥(context)으로부터 답을 찾을 수 없다면 '모른다'고 말하세요."
    " 반드시 한국어로 대답하세요.\n\n"
    "질문: {question}\n문맥: {context}\n답변:"
)

# -----------------------------
# 체인 구성
# -----------------------------
# 원본에서는 ChatOpenAI()에 모델명이 지정되지 않았음.
# → 명시적으로 model="gpt-4o-mini" 지정 (원하는 모델명으로 바꾸면 됨).
# RunnablePassthrough()는 그대로 사용 (입력값을 question으로 전달).
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")  
    | StrOutputParser()
)

# -----------------------------
# 질문 처리 함수
# -----------------------------
def chat_with_human(user_input: str) -> str:
    return chain.invoke(user_input)

# -----------------------------
# 메인 루프
# -----------------------------
# 원본에서는 단순 while문으로 작성되어 있어 Ctrl+C 종료 시 에러 발생 가능.
# → try/except KeyboardInterrupt 추가로 종료를 좀 더 깔끔하게 처리.
try:
    while True:
        user_input = input("User > ")
        if user_input.lower() == "quit":
            break
        print(f" AI > {chat_with_human(user_input)}")
except KeyboardInterrupt:
    print("\n종료합니다.")