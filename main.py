from dotenv import load_dotenv
from chat_assistant import MultiDatabaseChatAssistant

# 加载环境变量
load_dotenv()

def main():
    # 初始化多数据库聊天助手
    assistant = MultiDatabaseChatAssistant("./db/navigation/geojson.json")  # 指定位置数据文件路径

    print("USTC-Assistant")
    print(f"""Made by LLsdog and LkvO
    command:
        quit/exit : exit
        status : status of databases
        locations : view all available locations
        debug: <query> : view relevant document fragments""")
    while True:
        try:
            user_input = input("input>").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("llsdog good job")
                break

            if user_input.lower() == 'status':
                db_status = assistant.get_database_status()
                print("\ndatabases status:")
                for intent, info in db_status.items():
                    status_text = "✓ " if info['exists'] else "✗ "
                    type_text = f" [{info['type']}]"
                    print(f"{info['description']}: {status_text}{type_text}")
                    print(f"  path: {info['path']}")
                continue

            if user_input.lower() == 'locations':
                locations = assistant.get_available_locations()
                print(f"\n({len(locations)} available locations")
                for i, location in enumerate(locations, 1):
                    print(f"{i:2d}. {location}")
                continue

            if user_input.startswith('debug:'):
                question = user_input[6:].strip()
                intent = assistant.intent_classifier.classify(question)
                docs = assistant.search_similar_docs(question, intent)
                print(f"\ntype: {assistant.intent_classifier.get_intent_description(intent)}")
                print(f"relevant information:")
                for i, doc in enumerate(docs, 1):
                    print(f"information {i}:")
                    if isinstance(doc, str):
                        print(doc)
                    else:
                        print(doc.page_content)
                    print("-" * 50)
                continue

            # 正常问答
            result = assistant.query(user_input)
            print(f"\n【{result['intent_description']}】")
            print(f"answer: {result['answer']}")

            # 如果是导航结果，显示坐标信息
            if result.get('navigation_data'):
                nav_data = result['navigation_data']
                if nav_data['start']:
                    print(f"From: {nav_data['start']['name']} - {nav_data['start']['coords']}")
                if nav_data['end']:
                    print(f"To: {nav_data['end']['name']} - {nav_data['end']['coords']}")

        except KeyboardInterrupt:
            print("\nGoodBye.")
            break
        except Exception as e:
            print(f"Something wrong: {str(e)}")


if __name__ == "__main__":
    main()
