from openai import OpenAI
import os
from typing import List
from dotenv import load_dotenv
class DoubaoEmbeddings():
    load_dotenv()
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
