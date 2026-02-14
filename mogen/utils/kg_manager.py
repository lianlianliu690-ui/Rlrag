import json
import os

class KGManager:
    _instance = None
    _kg_map = None

    @classmethod
    def get_instance(cls, kg_path="/Dataset4D/public/mas-liu.lianlian/output/RAGesture/kg/beat_kg.json"):
        if cls._instance is None:
            cls._instance = KGManager(kg_path)
        return cls._instance

    def __init__(self, kg_path):
        if os.path.exists(kg_path):
            print(f"📚 Loading Knowledge Graph from {kg_path}")
            with open(kg_path, 'r') as f:
                self._kg_map = json.load(f)
        else:
            print(f"⚠️ KG file not found at {kg_path}, KG features disabled.")
            self._kg_map = {}

    def get_related_db_labels(self, query_word):
        """
        输入用户搜索词 (如 'colossal')
        返回数据库里对应的标签列表 (如 ['big', 'huge'])
        """
        query_word = query_word.lower().strip()
        return self._kg_map.get(query_word, [])