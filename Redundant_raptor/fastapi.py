from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum
from raptor import RetrievalAugmentation

class SubmitQueryRequest(BaseModel):
    query_text: str

app = FastAPI()

# CORS setup to allow Vercel frontend to communicate with FastAPI backend
origins = [
    "https://med-view-ai-system-frontend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # list of allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RA = RetrievalAugmentation()

@app.post("/submit_query")
def submit_query(request: SubmitQueryRequest):
    query_answer = RA.answer_question(request.query_text)
    return {"answer": query_answer}

handler = Mangum(app)
