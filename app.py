from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum
import pickle
import os
from dotenv import load_dotenv
import traceback

from raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.QAModels import GPT3TurboQAModel
from raptor.tree_retriever import TreeRetrieverConfig, TreeRetriever
from raptor.EmbeddingModels import OpenAIEmbeddingModel
from raptor.SummarizationModels import BaseSummarizationModel
from raptor.tree_builder import TreeBuilder, TreeBuilderConfig
from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig
from raptor.tree_structures import Node, Tree

app = FastAPI()
handler = Mangum(app)

class QueryRequest(BaseModel):
    query: str

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


with open("raptor/Freestyle_lite", "rb") as f:
    raptor_tree = pickle.load(f)

qa_model = GPT3TurboQAModel()

retriever_config = TreeRetrieverConfig(
    threshold=0.5,
    top_k=5,
    tokenizer=None,
    selection_mode="top_k",
    embedding_model=OpenAIEmbeddingModel()
)

config = RetrievalAugmentationConfig(
    tree_retriever_config=retriever_config,
    qa_model=qa_model,
    embedding_model=None,
    summarization_model=None,
    tree_builder_type="cluster"
)

raptor = RetrievalAugmentation(config=config)
raptor.tree = raptor_tree

tree_retriever = TreeRetriever(retriever_config, raptor_tree)
raptor.retriever = tree_retriever

@app.post("/retrieve")
def retrieve_from_query(request: QueryRequest):

    print("Query:", request.query)
    answer = raptor.answer_question(request.query)
    return {"answer": answer}
