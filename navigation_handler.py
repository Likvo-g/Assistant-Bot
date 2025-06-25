import os
import json
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from difflib import get_close_matches

class MapNavigationHandler:
    def __init__(self, locations_json_path= "./db/navigation/geojson.json"):
        self.locations_json_path = locations_json_path
        self.locations_data = {}
        self.location_names = []
        self._load_locations()

        # 初始化LLM用于位置提取
        self.llm = ChatOpenAI(
            openai_api_key=os.environ['API_KEY'],
            openai_api_base=os.environ['BASE_URL'],
            model_name=os.environ['MODEL']
        )

    def _load_locations(self):
        # 从geojson中获取位置信息
        try:
            with open(self.locations_json_path, 'r', encoding='utf-8') as f:
                locations_list = json.load(f)

            # 在location_names中存储名称,在locations[name]中存储相应的位置坐标
            for location in locations_list:
                name = location['name']
                self.locations_data[name] = location['coords']
                self.location_names.append(name)

            print(f" {len(self.locations_data)} locations loaded.")

        except Exception:
            print(f"warning: failed to load locations")

    def extract_locations_with_llm(self, query: str) -> Dict[str, Optional[str]]:
        # 直接使用大模型提取出发点和目标点, 当然提示词也是大模型写的
        available_locations = ", ".join(self.location_names[:20])  # 限制显示前20个位置

        prompt = f"""
        你是一个导航助手，需要从用户的导航请求中提取出发点和目的地。

        可用的校园位置包括（但不限于）：{available_locations}

        用户问题："{query}"

        请分析用户的导航需求，提取出发点和目的地。

        输出格式（JSON）：
        {{
            "start_location": "出发点名称或null",
            "end_location": "目的地名称",
            "navigation_type": "explicit"
        }}

        注意：
        1. 如果用户没有明确指定出发点，start_location设为null
        2. 目的地必须明确识别
        3. 位置名称要与可用位置列表匹配，如果不完全匹配，请选择最相似的
        4. 只输出JSON，不要其他内容
        """

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # 解析content的json文件
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            # 加载解析的json
            result = json.loads(content)
            return result

        except Exception:
            print(f"the llm failed to analyse locations")
            return {"start_location": None, "end_location": None, "navigation_type": "error"}

    def fuzzy_match_location(self, location_name: str, threshold: float = 0.3) -> Optional[str]:
        # 模糊名称匹配
        if not location_name:
            return None
        # 精确匹配
        if location_name in self.locations_data:
            return location_name
        # 模糊匹配
        matches = get_close_matches(location_name, self.location_names, n=1, cutoff=threshold)
        if matches:
            return matches[0]
        # 包含匹配
        for name in self.location_names:
            if location_name in name or name in location_name:
                return name
        return None

    def get_coordinates(self, location_name: str) -> Optional[List[float]]:
        # 获取位置坐标
        matched_name = self.fuzzy_match_location(location_name)
        if matched_name:
            return self.locations_data[matched_name]
        return None

    def process_navigation_request(self, query: str) -> Dict:
        # 1. 使用LLM提取位置信息
        extracted_info = self.extract_locations_with_llm(query)

        # 如果获取失败,直接返回
        if extracted_info["navigation_type"] == "error":
            return {
                "type": "navigation",
                "status": "error",
                "message": "failed analysis",
                "data": None
            }

        start_location = extracted_info.get("start_location")
        end_location = extracted_info.get("end_location")

        # 2. 获取坐标信息
        result_data = {
            "start": None,
            "end": None,
            "start_name": start_location,
            "end_name": end_location
        }
        # 处理起点
        if start_location:
            start_coords = self.get_coordinates(start_location)
            if start_coords:
                result_data["start"] = {
                    "name": self.fuzzy_match_location(start_location),
                    "coords": start_coords
                }
        # 处理终点
        if end_location:
            end_coords = self.get_coordinates(end_location)
            if end_coords:
                result_data["end"] = {
                    "name": self.fuzzy_match_location(end_location),
                    "coords": end_coords
                }
            else:
                return {
                    "type": "navigation",
                    "status": "error",
                    "message": f"cannot find location：{end_location}",
                    "data": None
                }
        else:
            return {
                "type": "navigation",
                "status": "error",
                "message": "cannot recognize locations, plz check",
                "data": None
            }

        return {
            "type": "navigation",
            "status": "success",
            "message": "get navigation information successfully",
            "data": result_data
        }
