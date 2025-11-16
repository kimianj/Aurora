"""
Question-Answering System for Member Data
Uses RAG (Retrieval-Augmented Generation) to answer questions about member messages.
"""

import os
import json
import logging
import re
from typing import List, Optional
from datetime import datetime

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Member Data QA System",
    description="Question-answering system for member messages",
    version="1.0.0"
)

# Configuration
API_BASE_URL = "http://november7-730026606190.europe-west1.run.app"
MESSAGES_ENDPOINT = f"{API_BASE_URL}/messages/"


# Global variables for cached data
messages_cache: List[dict] = []
embeddings_cache: Optional[np.ndarray] = None
embedding_model: Optional[SentenceTransformer] = None


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


def fetch_all_messages() -> List[dict]:
    """Fetch up to 100 messages from the API (pagination beyond 100 is blocked)."""
    logger.info("Fetching messages from API...")

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(MESSAGES_ENDPOINT, params={"offset": 0, "limit": 100})
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        logger.info(f"Fetched {len(items)} messages")
        return items


def initialize_embeddings():
    """Initialize the embedding model and generate embeddings for all messages."""
    global embedding_model, embeddings_cache, messages_cache
    
    logger.info("Initializing embedding model...")
    # Use a lightweight, fast model for embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if not messages_cache:
        messages_cache = fetch_all_messages()
    
    logger.info("Generating embeddings for messages...")
    # Create text representations of messages for embedding
    message_texts = []
    for msg in messages_cache:
        # Combine user name and message for better context
        text = f"{msg.get('user_name', '')}: {msg.get('message', '')}"
        message_texts.append(text)
    
    embeddings_cache = embedding_model.encode(message_texts, show_progress_bar=True)
    logger.info(f"Generated embeddings: shape {embeddings_cache.shape}")


def find_relevant_messages(question: str, top_k: int = 5) -> List[dict]:
    """Find the most relevant messages for a given question using cosine similarity."""
    if embedding_model is None or embeddings_cache is None:
        raise HTTPException(status_code=503, detail="Embeddings not initialized")
    
    # Generate embedding for the question
    question_embedding = embedding_model.encode([question])
    
    # Calculate cosine similarity
    similarities = np.dot(embeddings_cache, question_embedding.T).flatten()
    
    # Get top-k most similar messages
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    relevant_messages = []
    for idx in top_indices:
        relevant_messages.append({
            "message": messages_cache[idx],
            "similarity": float(similarities[idx])
        })
    
    return relevant_messages


def generate_answer(question: str, relevant_messages: List[dict]) -> str:
    """Generate an answer using intelligent pattern matching - no external APIs needed."""
    # Use the improved extraction function that works entirely locally
    return extract_answer_simple(question, relevant_messages)


def extract_answer_simple(question: str, relevant_messages: List[dict]) -> str:
    """Intelligent answer extraction using pattern matching - works entirely locally, no external APIs."""
    question_lower = question.lower()
    
    # Extract person name from question
    person_name = None
    for msg in messages_cache:
        name = msg.get('user_name', '')
        if name and name.lower() in question_lower:
            person_name = name
            break
    
    # Filter messages by person if specified
    if person_name:
        relevant_messages = [
            item for item in relevant_messages
            if item["message"].get('user_name', '').lower() == person_name.lower()
        ]
    
    if not relevant_messages:
        return "I don't have enough information to answer this question."
    
    # Pattern-based answer extraction
    # Check for "when" questions (dates/trips)
    if any(word in question_lower for word in ['when', 'date', 'time', 'trip', 'planning']):
        results = []
        for item in relevant_messages:
            msg = item["message"]
            message_text = msg.get('message', '').lower()
            timestamp = msg.get('timestamp', '')
            user_name = msg.get('user_name', 'Unknown')
            
            # Extract date from message text if mentioned
            date_patterns = [
                r'(?:this|next|on)\s+(?:friday|saturday|sunday|monday|tuesday|wednesday|thursday)',
                r'(?:this|next)\s+(?:week|month)',
                r'\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)',
                r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
            ]
            
            found_date = None
            for pattern in date_patterns:
                match = re.search(pattern, message_text, re.IGNORECASE)
                if match:
                    found_date = match.group(0)
                    break
            
            # Use timestamp if no date in message
            if not found_date and timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    found_date = dt.strftime('%B %d, %Y')
                except:
                    pass
            
            if found_date or timestamp:
                location = None
                # Try to extract location
                location_keywords = ['london', 'paris', 'tokyo', 'new york', 'dubai', 'monaco', 'milan', 'bangkok', 'hong kong']
                for loc in location_keywords:
                    if loc in message_text:
                        location = loc.title()
                        break
                
                result = f"{user_name}"
                if location:
                    result += f" is planning a trip to {location}"
                if found_date:
                    result += f" for {found_date}"
                elif timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        result += f" (message from {dt.strftime('%B %d, %Y')})"
                    except:
                        pass
                result += f": {msg.get('message', '')}"
                results.append(result)
        
        if results:
            return results[0] if len(results) == 1 else " | ".join(results[:2])
    
    # Check for "how many" questions
    if 'how many' in question_lower:
        # Extract what they're asking about (cars, tickets, etc.)
        subject = None
        subjects = ['car', 'cars', 'ticket', 'tickets', 'seat', 'seats', 'person', 'people']
        for subj in subjects:
            if subj in question_lower:
                subject = subj
                break
        
        counts = []
        for item in relevant_messages:
            msg = item["message"]
            msg_text = msg.get('message', '')
            user_name = msg.get('user_name', 'Unknown')
            
            # Look for numbers in the message
            numbers = re.findall(r'\d+', msg_text)
            if numbers and (not subject or subject.rstrip('s') in msg_text.lower()):
                # Try to find the number associated with the subject
                if subject:
                    # Look for patterns like "2 cars", "three tickets", etc.
                    pattern = rf'(\d+)\s+{subject}'
                    match = re.search(pattern, msg_text.lower())
                    if match:
                        counts.append(f"{user_name} has {match.group(1)} {subject}: {msg_text}")
                    else:
                        # Just use the first number found
                        counts.append(f"{user_name}: {msg_text}")
                else:
                    counts.append(f"{user_name}: {msg_text}")
        
        if counts:
            # Try to sum up if multiple mentions
            total = 0
            for count_str in counts:
                numbers = re.findall(r'\d+', count_str)
                if numbers:
                    total += sum(int(n) for n in numbers)
            
            if total > 0 and subject:
                return f"{person_name or 'The user'} has {total} {subject}."
            return " | ".join(counts[:2])
    
    # Check for "what" or "list" questions (preferences, restaurants, favorites)
    if any(word in question_lower for word in ['what', 'list', 'favorite', 'favourite', 'prefer', 'preference', 'restaurant']):
        items = []
        for item in relevant_messages[:10]:  # Check more messages for lists
            msg = item["message"]
            msg_text = msg.get('message', '').lower()
            user_name = msg.get('user_name', 'Unknown')
            
            # Extract restaurant names (common patterns)
            restaurant_patterns = [
                r'(?:at|to|reservation at|book|reserve)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:restaurant|dining|dinner)',
            ]
            
            restaurants = []
            for pattern in restaurant_patterns:
                matches = re.findall(pattern, msg.get('message', ''))
                restaurants.extend(matches)
            
            if restaurants:
                items.append(f"{user_name}'s favorite restaurants: {', '.join(set(restaurants))}")
            elif any(word in msg_text for word in ['restaurant', 'dining', 'dinner', 'favorite', 'prefer']):
                # Extract the restaurant name or preference
                items.append(f"{user_name}: {msg.get('message', '')}")
        
        if items:
            return " | ".join(items[:3])
    
    # Default: return top relevant messages with context
    answers = []
    for item in relevant_messages[:3]:
        msg = item["message"]
        user_name = msg.get('user_name', 'Unknown')
        message = msg.get('message', '')
        timestamp = msg.get('timestamp', '')
        
        answer = f"{user_name}: {message}"
        if timestamp and 'when' in question_lower:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                answer += f" (on {dt.strftime('%B %d, %Y')})"
            except:
                pass
        
        answers.append(answer)
    
    return " | ".join(answers)


def analyze_data_insights() -> dict:
    """Analyze the dataset for anomalies and inconsistencies."""
    if not messages_cache:
        return {}
    
    insights = {
        "total_messages": len(messages_cache),
        "unique_users": len(set(msg.get('user_id') for msg in messages_cache)),
        "date_range": {},
        "anomalies": []
    }
    
    # Extract dates
    timestamps = []
    for msg in messages_cache:
        ts = msg.get('timestamp')
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            except:
                pass
    
    if timestamps:
        insights["date_range"] = {
            "earliest": min(timestamps).isoformat(),
            "latest": max(timestamps).isoformat()
        }
    
    # Check for missing data
    missing_user_names = sum(1 for msg in messages_cache if not msg.get('user_name'))
    missing_messages = sum(1 for msg in messages_cache if not msg.get('message'))
    missing_timestamps = sum(1 for msg in messages_cache if not msg.get('timestamp'))
    
    if missing_user_names > 0:
        insights["anomalies"].append(f"{missing_user_names} messages missing user_name")
    if missing_messages > 0:
        insights["anomalies"].append(f"{missing_messages} messages missing message content")
    if missing_timestamps > 0:
        insights["anomalies"].append(f"{missing_timestamps} messages missing timestamp")
    
    # Check for encoding issues (like the "MAller" we saw - should be "Müller")
    # Look for replacement characters () which indicate encoding problems
    encoding_issues = 0
    for msg in messages_cache:
        name = str(msg.get('user_name', ''))
        # Check for replacement character (U+FFFD) or other encoding artifacts
        if '\ufffd' in name or (name and not name.isprintable()):
            encoding_issues += 1
    if encoding_issues > 0:
        insights["anomalies"].append(f"{encoding_issues} messages with potential encoding issues in user names")
    
    # Check for future dates (anomaly)
    now = datetime.now()
    future_dates = sum(1 for ts in timestamps if ts > now)
    if future_dates > 0:
        insights["anomalies"].append(f"{future_dates} messages with timestamps in the future")
    
    return insights


@app.on_event("startup")
async def startup_event():
    try:
        initialize_embeddings()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.warning("Continuing without initialized embeddings; endpoints may be limited")



@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Member Data QA System",
        "endpoints": {
            "/ask": "POST - Ask a question (body: {'question': '...'})",
            "/health": "GET - Health check",
            "/insights": "GET - Data insights and anomalies"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "messages_loaded": len(messages_cache),
        "embeddings_ready": embeddings_cache is not None
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Answer a question about member data."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Find relevant messages
        relevant_messages = find_relevant_messages(request.question, top_k=5)
        # If similarity is too low, don't guess
        similarities = [item["similarity"] for item in relevant_messages]
        if max(similarities) < 0.25:  # threshold can be tuned
            return AnswerResponse(answer="I don't have enough information to answer this question.")

        
        if not relevant_messages:
            return AnswerResponse(answer="I don't have enough information to answer this question.")

        
        # Generate answer
        answer = generate_answer(request.question, relevant_messages)
        
        return AnswerResponse(answer=answer)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/ask")
async def ask_question_get(question: str = Query(..., description="The question to answer")):
    """Answer a question about member data (GET endpoint for convenience)."""
    request = QuestionRequest(question=question)
    return await ask_question(request)


@app.get("/insights")
async def get_insights():
    """Get data insights and anomalies."""
    insights = analyze_data_insights()
    return insights


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

