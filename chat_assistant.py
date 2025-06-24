from langchain_community.vectorstores import Chroma
from openai import OpenAI
import os
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 构建豆包Embeddings（用于查询时的向量化）
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


class ChatAssistant:
    def __init__(self, vectorstore_path=os.environ['CLASSES_PATH']):
        self.vectorstore_path = vectorstore_path
        self.embeddings = DoubaoEmbeddings()
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self._setup()

    def _setup(self):
        """初始化向量数据库和问答链"""
        print("正在加载向量数据库...")

        # 从本地加载向量数据库
        self.vectorstore = Chroma(
            persist_directory=self.vectorstore_path,
            embedding_function=self.embeddings
        )

        # 从vectorstore中检索相关文档对象
        self.retriever = self.vectorstore.as_retriever()

        # 构建豆包问答模板
        template = """
        以下问题基于提供的 context，分析问题并给出合理的建议回答：
        <context>
        {context}
        </context>
        Question: {input}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 调用豆包OpenAI
        llm = ChatOpenAI(
            openai_api_key=os.environ['API_KEY'],
            openai_api_base=os.environ['BASE_URL'],
            model_name=os.environ['MODEL']
        )

        # 构建豆包问答链
        self.rag_chain = (
                {"context": self.retriever, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        print("问答系统初始化完成！")

    def query(self, question: str) -> str:
        """查询问题并返回答案"""
        try:
            return self.rag_chain.invoke(question)
        except Exception as e:
            return f"查询出错: {str(e)}"

    def search_similar_docs(self, question: str, k: int = 3):
        """搜索相似文档（用于调试）"""
        docs = self.retriever.get_relevant_documents(question)
        return docs[:k]


def main():
    # 检查向量数据库是否存在
    vectorstore_path = os.environ['CLASSES_PATH']
    if not os.path.exists(vectorstore_path):
        print(f"向量数据库不存在: {vectorstore_path}")
        print("请先运行 generate_vectorstore.py 生成向量数据库")
        return

    # 初始化聊天助手
    assistant = ChatAssistant(vectorstore_path)

    print("=== USTC-Assistant ===")
    print("输入 'quit' 或 'exit' 退出程序")
    print("输入 'debug: <问题>' 查看相关文档片段")

    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break

            if user_input.startswith('debug:'):
                question = user_input[6:].strip()
                docs = assistant.search_similar_docs(question)
                print(f"\n=== 相关文档片段 ===")
                for i, doc in enumerate(docs, 1):
                    print(f"片段 {i}:")
                    print(doc.page_content)
                    print("-" * 50)
            else:
                print("正在思考...")
                answer = assistant.query(user_input)
                print(f"\n回答: {answer}")

        except KeyboardInterrupt:
            print("\n程序被中断，再见！")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
