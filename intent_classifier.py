import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Optional

load_dotenv()


class IntentClassifier:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ['INTENT_API_KEY'],
            base_url=os.environ['INTENT_BASE_URL']
        )

        # 三种意图及其对应的数据库路径
        self.intent_config = {
            'campus_navigation': {
                'vectorstore_path': os.environ.get('NAVIGATION_PATH', './db/navigation'),
                'description': '校园导航'
            },
            'campus_information': {
                'vectorstore_path': os.environ.get('CAMPUS_PATH', './db/campus'),
                'description': '校园信息'
            },
            'course_selection': {
                'vectorstore_path': os.environ.get('COURSE_PATH', './db/course'),
                'description': '选课相关'
            }
        }

    def classify(self, question: str) -> str:
        """分类用户问题"""
        try:
            completion = self.client.chat.completions.create(
                model=os.environ['INTENT_MODEL'],
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个意图判断者,需要将用户的话分为三种类型,并输出对应结果:'campus_navigation'(校园导航)、'campus_information'(校园信息)、'course_selection'(选课相关)。仅需输出这三个选项名之一,不需要其他输出"
                    },
                    {"role": "user", "content": question}
                ]
            )

            result = completion.choices[0].message.content.strip()

            # 清理结果，确保返回标准格式
            for intent in self.intent_config.keys():
                if intent in result:
                    return intent

            # 如果无法匹配，返回默认意图
            print(f"警告: 无法识别意图 '{result}', 使用默认意图 'campus_information'")
            return 'campus_information'

        except Exception as e:
            print(f"意图分类出错: {str(e)}, 使用默认意图 'campus_information'")
            return 'campus_information'

    def get_vectorstore_path(self, intent: str) -> str:
        """根据意图获取对应的向量数据库路径"""
        return self.intent_config.get(intent, {}).get('vectorstore_path', os.environ['CAMPUS_PATH'])

    def get_intent_description(self, intent: str) -> str:
        """获取意图描述"""
        return self.intent_config.get(intent, {}).get('description', '未知意图')

    def get_all_intents(self) -> Dict:
        """获取所有意图配置"""
        return self.intent_config
