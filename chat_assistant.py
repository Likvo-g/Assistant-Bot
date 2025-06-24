from langchain_community.vectorstores import Chroma
from openai import OpenAI
import os
from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from intent_classifier import IntentClassifier

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


class MultiDatabaseChatAssistant:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.embeddings = DoubaoEmbeddings()
        self.vectorstores = {}  # 缓存已加载的向量数据库
        self.retrievers = {}  # 缓存已创建的检索器
        self.rag_chains = {}  # 缓存已创建的RAG链
        self.llm = None
        self._setup_llm()

    def _setup_llm(self):
        """初始化语言模型"""
        self.llm = ChatOpenAI(
            openai_api_key=os.environ['API_KEY'],
            openai_api_base=os.environ['BASE_URL'],
            model_name=os.environ['MODEL']
        )

    def _load_vectorstore(self, vectorstore_path: str, intent: str):
        """加载指定路径的向量数据库"""
        if intent in self.vectorstores:
            return self.vectorstores[intent]

        if not os.path.exists(vectorstore_path):
            print(f"警告: 向量数据库不存在: {vectorstore_path}")
            return None

        try:
            print(f"正在加载 {self.intent_classifier.get_intent_description(intent)} 数据库...")
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=self.embeddings
            )
            self.vectorstores[intent] = vectorstore
            return vectorstore
        except Exception as e:
            print(f"加载向量数据库失败: {str(e)}")
            return None

    def _get_retriever(self, intent: str):
        """获取指定意图的检索器"""
        if intent in self.retrievers:
            return self.retrievers[intent]

        vectorstore_path = self.intent_classifier.get_vectorstore_path(intent)
        vectorstore = self._load_vectorstore(vectorstore_path, intent)

        if vectorstore is None:
            return None

        retriever = vectorstore.as_retriever()
        self.retrievers[intent] = retriever
        return retriever

    def _get_rag_chain(self, intent: str):
        """获取指定意图的RAG链"""
        if intent in self.rag_chains:
            return self.rag_chains[intent]

        retriever = self._get_retriever(intent)
        if retriever is None:
            return None

        # 根据意图定制不同的提示模板
        templates = {
            'campus_navigation': """
            你是USTC校园导航助手。基于提供的校园地图和位置信息，为用户提供准确的导航指引。

            <context>
            {context}
            </context>

            问题: {input}

            请提供详细的路线指引，包括具体的地标、方向和距离信息。
            """,

            'campus_information': """
            你是USTC校园信息助手。基于提供的校园相关信息，回答用户关于学校设施、服务、规定等问题。

            <context>
            {context}
            </context>

            问题: {input}

            请提供准确、详细的校园信息，包括时间、地点、联系方式等具体细节。
            """,

            'course_selection': """
            你是USTC选课助手。基于提供的课程信息和选课规则，帮助用户解决选课相关问题。

            <context>
            {context}
            </context>

            问题: {input}

            请提供具体的选课指导，包括操作步骤、注意事项和相关政策。
            """
        }

        template = templates.get(intent, templates['campus_information'])
        prompt = ChatPromptTemplate.from_template(template)

        # 构建RAG链
        rag_chain = (
                {"context": retriever, "input": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        self.rag_chains[intent] = rag_chain
        return rag_chain

    def query(self, question: str) -> Dict[str, str]:
        """查询问题并返回答案"""
        try:
            # 1. 意图分类
            print("正在分析问题类型...")
            intent = self.intent_classifier.classify(question)
            intent_desc = self.intent_classifier.get_intent_description(intent)
            print(f"问题类型: {intent_desc}")

            # 2. 获取对应的RAG链
            rag_chain = self._get_rag_chain(intent)
            if rag_chain is None:
                return {
                    "intent": intent,
                    "intent_description": intent_desc,
                    "answer": f"抱歉，{intent_desc}数据库暂时不可用。",
                    "status": "error"
                }

            # 3. 生成答案
            print("正在生成答案...")
            answer = rag_chain.invoke(question)

            return {
                "intent": intent,
                "intent_description": intent_desc,
                "answer": answer,
                "status": "success"
            }

        except Exception as e:
            return {
                "intent": "unknown",
                "intent_description": "未知",
                "answer": f"查询出错: {str(e)}",
                "status": "error"
            }

    def search_similar_docs(self, question: str, intent: str = None, k: int = 3):
        """搜索相似文档（用于调试）"""
        if intent is None:
            intent = self.intent_classifier.classify(question)

        retriever = self._get_retriever(intent)
        if retriever is None:
            return []

        docs = retriever.get_relevant_documents(question)
        return docs[:k]

    def get_database_status(self) -> Dict:
        """获取所有数据库的状态"""
        status = {}
        for intent, config in self.intent_classifier.get_all_intents().items():
            path = config['vectorstore_path']
            status[intent] = {
                'description': config['description'],
                'path': path,
                'exists': os.path.exists(path),
                'loaded': intent in self.vectorstores
            }
        return status


def main():
    # 初始化多数据库聊天助手
    assistant = MultiDatabaseChatAssistant()

    print("=== USTC 智能助手 ===")
    print("我可以帮助您解决校园导航、校园信息和选课相关问题")
    print("输入 'quit' 或 'exit' 退出程序")
    print("输入 'status' 查看数据库状态")
    print("输入 'debug: <问题>' 查看相关文档片段")

    # 显示数据库状态
    db_status = assistant.get_database_status()
    print("\n=== 数据库状态 ===")
    for intent, info in db_status.items():
        status_text = "✓" if info['exists'] else "✗"
        print(f"{status_text} {info['description']}: {info['path']}")

    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break

            if user_input.lower() == 'status':
                db_status = assistant.get_database_status()
                print("\n=== 数据库状态 ===")
                for intent, info in db_status.items():
                    status_text = "✓ 可用" if info['exists'] else "✗ 不可用"
                    loaded_text = " (已加载)" if info['loaded'] else ""
                    print(f"{info['description']}: {status_text}{loaded_text}")
                    print(f"  路径: {info['path']}")
                continue

            if user_input.startswith('debug:'):
                question = user_input[6:].strip()
                intent = assistant.intent_classifier.classify(question)
                docs = assistant.search_similar_docs(question, intent)
                print(f"\n=== 问题类型: {assistant.intent_classifier.get_intent_description(intent)} ===")
                print(f"=== 相关文档片段 ===")
                for i, doc in enumerate(docs, 1):
                    print(f"片段 {i}:")
                    print(doc.page_content)
                    print("-" * 50)
                continue

            # 正常问答
            result = assistant.query(user_input)
            print(f"\n【{result['intent_description']}】")
            print(f"回答: {result['answer']}")

        except KeyboardInterrupt:
            print("\n程序被中断，再见！")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()