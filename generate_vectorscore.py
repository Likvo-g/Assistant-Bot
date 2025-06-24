import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
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
    loader = TextLoader("./data/ustcGuide.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    embeddings = DoubaoEmbeddings()
    persist_directory = "./db/campus_information"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"向量数据库已保存到: {persist_directory}")
    return vectorstore

# 生成数据库并保存到./db/25spring_classes
def generate_vectorscore_xlsx():
    try:
        # 使用pandas读取Excel文件
        df = pd.read_excel("./data/2025spring.xlsx")
        # 将每一行转换为Document对象
        documents = []
        for index, row in df.iterrows():
            course_info_parts = []

            for column in df.columns:
                value = row[column]
                # 跳过空值
                if pd.notna(value) and str(value).strip():
                    course_info_parts.append(f"{column}: {value}")

            course_text = "\n".join(course_info_parts)

            doc = Document(
                page_content=course_text,
                metadata={
                    "source": "2025spring.xlsx",
                    "row_index": index,
                    "course_id": str(row.get("课堂号", "")),
                    "course_name": str(row.get("课程名", "")),
                    "instructor": str(row.get("授课教师", "")),
                }
            )
            documents.append(doc)

        print(f"成功创建 {len(documents)} 个课程文档")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 增大chunk_size以保持课程信息完整性
            chunk_overlap=100,
            separators=["\n\n", "\n", "，", "。", " ", ""]
        )
        chunks = documents
        # 可选择分割文档处理
        # chunks = text_splitter.split_documents(documents)
        print(f"amount: {len(chunks)}")
        # 创建向量化模型
        embeddings = DoubaoEmbeddings()
        # 创建向量数据库
        persist_directory = "./db/course"
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        print(f"向量数据库已保存到: {persist_directory}")
        return vectorstore

    except FileNotFoundError:
        return None
    except Exception as e:
        return None


if __name__ == "__main__":
    #generate_vectorstore_data_txt()
    generate_vectorscore_xlsx()
