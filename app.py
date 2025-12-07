# Install required packages before running:
# pip install fastapi uvicorn transformers sqlite3

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
import sqlite3
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Initialize Q&A pipeline (using a pre-trained transformer model)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Set up SQLite DB connection and create logs table if not exists
conn = sqlite3.connect('chatbot_logs.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_question TEXT,
                    bot_answer TEXT,
                    timestamp TEXT)''')
conn.commit()

# Sample FAQ context (could be expanded or replaced with real data)
faq_context = """
Welcome to our customer support. We offer 24/7 support on all products. For shipping information, 
orders take 3-5 business days. Returns are accepted within 30 days. Contact email: support@example.com.
"""

# Input structure for API
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_bot(query: Query):
    # Use NLP model to answer question based on context
    result = qa_pipeline(question=query.question, context=faq_context)
    answer = result['answer']

    # Log interaction to SQLite DB
    cursor.execute("INSERT INTO logs (user_question, bot_answer, timestamp) VALUES (?, ?, ?)",
                   (query.question, answer, datetime.now().isoformat()))
    conn.commit()

    # Return chatbot answer
    return {"answer": answer}