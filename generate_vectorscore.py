from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_community.document_loaders import TextLoader, UnstructuredExcelLoader
import os
from typing import List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# 构建豆包Embeddings
class DoubaoEmbeddings():
    client: OpenAI = None
    api_key: str = os.environ['EMBEDDING_API_KEY']
    model: str = os.environ['EMBEDDING_MODEL']

    def __init__(self, **data: any):
        super().__init__(**data)
        if self.api_key == "":
            self.api_key = os.environ['EMBEDDING_API_KEY']

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.environ['EMBEDDING_BASE_URL']
        )

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.
        Args:
            texts (str): 要生成 embedding 的文本.
        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """
        embeddings = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    class Config:
        arbitrary_types_allowed = True

# 生成数据库并保存到db/campus_information
def generate_vectorstore_data_txt():
    print("开始加载文档...")
    # 加载文本
    loader = TextLoader("./data/data.txt", encoding="utf-8")
    documents = loader.load()

    print("开始切割文档...")
    # 切割文本
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"文档切割完成，共 {len(chunks)} 个片段")

    print("开始生成向量数据库...")
    # 生成豆包Embeddings
    embeddings = DoubaoEmbeddings()

    # 指定本地存储路径
    persist_directory = "./db/campus_information"

    # 存储向量后的数据到本地
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"向量数据库已保存到: {persist_directory}")
    return vectorstore

# 生成数据库并保存到./db/25spring_classes
def generate_vectorscore_xlsx():
    loader = UnstructuredExcelLoader("./data/2025spring.xlsx")
    documents = loader.load()

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap = 50
    )
    chunks = text_splitter.split_documents(documents)

    # 创建向量化模型
    embeddings = DoubaoEmbeddings()
    # 创建向量数据库
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./db/classes_information"
    )

    return vectorstore


if __name__ == "__main__":
    generate_vectorstore_data_txt()
    generate_vectorscore_xlsx()
