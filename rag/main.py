import os
groq_api_key = os.environ["GROQ_API_KEY"]


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 仮の import。実際には langchain_groq のドキュメントに従ってインストール＆import
from langchain_groq import ChatGroq

def setup_vectorstore_from_pdf(pdf_path: str, persist_directory: str = "./chroma_db"):
    """
    指定した PDF のテキストを分割し、埋め込みを作成し、Chroma のベクトルストアに保存します。
    persist_directory に既存の DB があれば継続利用、なければ新規作成します。
    """
    # 1. PDF を読み込む
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. テキストを分割する
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # 3. Embeddings を作成（ここでは Hugging Face のモデル例）
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Chroma へ保存（あるいは既存 DB があれば読み込んで再構築しないように工夫）
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectordb.persist()
    else:
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
    
    return vectordb

def main():
    pdf_path = "example.pdf"  # 実際の PDF パスを指定
    vectordb = setup_vectorstore_from_pdf(pdf_path)

    # Groq 用の LLM（ChatGroq インスタンス）を用意
    # 実際には適切な引数や認証方法などを設定してください
    groq_llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        groq_api_key=groq_api_key,
        # 以下は仮想のパラメータ例
        # api_key=os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY"),
        # endpoint_url="https://api.groq.example.com/v1/completions"
    )

    # RetrievalQA でチェーンを構築（chain_type="stuff" はシンプルな合体形式の例）
    qa_chain = RetrievalQA.from_chain_type(
        llm=groq_llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3})
    )

    # テスト的にクエリを投げてみる
    query = "PDF の中で解説されている主要なトピックは何ですか？"
    result = qa_chain.run(query)
    
    print("=== AI の回答 ===")
    print(result)

if __name__ == "__main__":
    main()

