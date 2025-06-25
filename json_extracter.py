import json
from fuzzywuzzy import fuzz, process
import jieba

# 将geojson格式转换为json格式,只保留地点名称和经纬度,再将经纬度传给高德地图的API
# 使用了openStreet的数据,然后将osm数据转换为json格式.

def extract_name_and_coords(geojson_data):
    search_items = []

    # 如果没有feature 返回空列表
    for feature in geojson_data.get('features', []):
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})

        # 提取名称和坐标
        name = properties.get('name', '').strip()
        if not name:
            continue

        if geometry.get('type') == 'Point':
            coords = geometry.get('coordinates', [])
            if len(coords) != 2:
                continue
        # 存储坐标和名称
        search_items.append({
            'name' : name,
            'coords': coords, # [lng, lat]
        })

        with open('./db/navigation/geojson.json', 'w', encoding='utf-8') as file:
            # 允许保存非ascii字符,缩进设置为2
            json.dump(search_items, file, ensure_ascii=False, indent=2)

with open('./data/map.json', 'r', encoding='utf-8') as file:
    geojson_data = json.load(file)
    extract_name_and_coords(geojson_data)