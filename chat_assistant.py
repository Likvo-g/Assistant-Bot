from langchain_community.vectorstores import Chroma
import os
from typing import List, Dict, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from embeddings import DoubaoEmbeddings
from intent_classifier import IntentClassifier
from navigation_handler import MapNavigationHandler

class MultiDatabaseChatAssistant:
    def __init__(self, locations_json_path):
        self.intent_classifier = IntentClassifier()
        self.embeddings = DoubaoEmbeddings()
        self.vectorstores = {}  # 缓存已加载的向量数据库
        self.retrievers = {}  # 缓存已创建的检索器
        self.rag_chains = {}  # 缓存已创建的RAG链
        self.llm = None

        self.navigation_handler = MapNavigationHandler(locations_json_path)

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
            print(f"vectorstore does not exist: {vectorstore_path}")
            return None

        try:
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=self.embeddings
            )
            self.vectorstores[intent] = vectorstore
            return vectorstore
        except Exception as e:
            print(f"failed to load vectorstore: {str(e)}")
            return None

    def _get_retriever(self, intent: str):
        # 根据意图获取检索器
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
        # 根据意图获取RAG链
        if intent in self.rag_chains:
            return self.rag_chains[intent]

        retriever = self._get_retriever(intent)
        if retriever is None:
            return None

        # 根据意图定制不同的提示模板
        templates = {
            'campus_information': """以下问题基于提供的上下文，分析问题并给出合理的建议回答：
<context>
{context}
</context>
Question: {input}
            """,

            'course_selection': """以下问题基于提供的上下文，分析问题并给出合理的建议回答：
<context>
{context}
</context>
Question: {input}
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

    def query(self, question: str) -> Dict:
        # 生成答案
        try:
            # 1. 意图分类
            intent = self.intent_classifier.classify(question)
            intent_desc = self.intent_classifier.get_intent_description(intent)
            print(f"type: {intent_desc}")

            # 2. 如果是导航意图，使用地图导航处理器
            if intent == 'campus_navigation':
                navigation_result = self.navigation_handler.process_navigation_request(question)

                if navigation_result["status"] == "success":
                    data = navigation_result["data"]
                    start_info = data["start"]
                    end_info = data["end"]

                    if start_info:
                        answer = f"导航路线规划：\n从 {start_info['name']} (坐标: {start_info['coords']}) \n到 {end_info['name']} (坐标: {end_info['coords']})\n\n请在地图上查看详细路线。"
                    else:
                        answer = f"目的地：{end_info['name']} (坐标: {end_info['coords']})\n\n请在地图上查看从您当前位置到目的地的路线。"

                    return {
                        "intent": intent,
                        "intent_description": intent_desc,
                        "answer": answer,
                        "status": "success",
                        "navigation_data": navigation_result["data"]  # 额外返回导航数据供前端使用
                    }
                else:
                    return {
                        "intent": intent,
                        "intent_description": intent_desc,
                        "answer": navigation_result["message"],
                        "status": "error",
                        "navigation_data": None
                    }

            # 其他意图使用原有的RAG处理
            rag_chain = self._get_rag_chain(intent)

            # 生成答案
            print("Thinking...")
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
                "intent_description": "unknown",
                "answer": f"query failed: {str(e)}",
                "status": "error"
            }

    def search_similar_docs(self, question: str, intent: str = None, k: int = 3):
        # 搜索相似文档
        if intent is None:
            intent = self.intent_classifier.classify(question)

        # 如果是导航意图，返回导航处理结果
        if intent == 'campus_navigation':
            navigation_result = self.navigation_handler.process_navigation_request(question)
            return [f"navigation result: {navigation_result}"]

        retriever = self._get_retriever(intent)
        if retriever is None:
            return []

        docs = retriever.get_relevant_documents(question)
        return docs[:k]

    def get_database_status(self) -> Dict:
        # 获取数据库状态
        status = {}
        for intent, config in self.intent_classifier.get_all_intents().items():
            if intent == 'campus_navigation':
                # 导航使用JSON文件而不是向量数据库
                status[intent] = {
                    'description': config['description'],
                    'path': self.navigation_handler.locations_json_path,
                    'exists': os.path.exists(self.navigation_handler.locations_json_path),
                    'loaded': len(self.navigation_handler.locations_data) > 0,
                    'type': 'json_locations'
                }
            else:
                path = config['vectorstore_path']
                status[intent] = {
                    'description': config['description'],
                    'path': path,
                    'exists': os.path.exists(path),
                    'loaded': intent in self.vectorstores,
                    'type': 'vectorstore'
                }
        return status

    def get_available_locations(self) -> List[str]:
        # 获取所有位置
        return self.navigation_handler.location_names
