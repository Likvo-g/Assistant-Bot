from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import importlib.util

# 创建应用
app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 确保处理OPTIONS预检请求
@app.options("/{path:path}")
async def options_handler(request: Request):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )


# 定义请求模型
class QueryRequest(BaseModel):
    question: str


# 动态导入旧/新版本的ChatAssistant
def get_chat_assistant():
    # 首先尝试导入新版本的助手
    try:
        # 检测是否存在 MultiDatabaseChatAssistant
        if os.path.exists("chat_assistant.py"):
            spec = importlib.util.spec_from_file_location("chat_assistant", "chat_assistant.py")
            chat_assistant_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(chat_assistant_module)

            # 检查模块中是否有MultiDatabaseChatAssistant类
            if hasattr(chat_assistant_module, "MultiDatabaseChatAssistant"):
                locations_json_path = os.environ.get("LOCATIONS_JSON_PATH", "./db/navigation/geojson.json")
                return chat_assistant_module.MultiDatabaseChatAssistant(locations_json_path)
            else:
                # 如果没有新类，则使用旧版本的ChatAssistant
                return chat_assistant_module.ChatAssistant()
    except Exception as e:
        print(f"加载新助手时出错: {str(e)}")

    # 回退到导入旧版本的助手
    try:
        import chat_assistant
        return chat_assistant.ChatAssistant()
    except Exception as e:
        print(f"加载助手时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"初始化聊天助手失败: {str(e)}")


# 初始化助手
assistant = get_chat_assistant()


# 修改旧版API端点，以便在导航情况下返回完整信息
@app.post("/predict")
async def predict(request: QueryRequest):
    try:
        # 检测assistant对象类型并相应处理
        if hasattr(assistant, "query"):
            # 新版本返回dict
            result = assistant.query(request.question)

            # 如果是导航意图，直接返回完整的结果对象
            if result.get("intent") == "campus_navigation":
                return result

            # 否则只返回答案部分
            return {"response": result["answer"]}
        else:
            # 旧版本直接返回字符串
            response = assistant.query(request.question)
            return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 新版API端点 - 返回完整的结果对象
@app.post("/v2/predict")
async def predict_v2(request: QueryRequest):
    try:
        # 检测assistant对象类型
        if hasattr(assistant, "query"):
            # 新版本返回完整结果
            result = assistant.query(request.question)
            return result
        else:
            # 旧版本包装成新格式
            response = assistant.query(request.question)
            return {
                "intent": "unknown",
                "intent_description": "Legacy API",
                "answer": response,
                "status": "success"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 获取可用位置（仅新版API支持）
@app.get("/v2/locations")
async def get_locations():
    try:
        if hasattr(assistant, "get_available_locations"):
            locations = assistant.get_available_locations()
            return {"locations": locations}
        else:
            return {"locations": [], "message": "此功能仅在v2版本可用"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 获取数据库状态（仅新版API支持）
@app.get("/v2/status")
async def get_database_status():
    try:
        if hasattr(assistant, "get_database_status"):
            status = assistant.get_database_status()
            return {"status": status}
        else:
            return {"status": {}, "message": "此功能仅在v2版本可用"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
