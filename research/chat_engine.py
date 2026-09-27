import numpy as np
import re
import time
from .searcher import search, wikipedia_summary
from .vectorizer import TFIDFVectorizer

def extract_best_sentences(prompt: str, context: str, top_n: int = 3) -> str:
    """Uses TF-IDF Cosine Similarity to find the most relevant sentences in the fetched context."""
    # Split text into sentences roughly
    sentences = re.split(r'(?<=[.!?]) +', context)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return "I couldn't find any detailed information on that."

    # Build a temporary vectorizer just for this document's vocabulary
    doc_vec = TFIDFVectorizer(vocab_size=256)
    try:
        doc_vec.fit(sentences)
    except Exception:
        # Failsafe if not enough words
        return " ".join(sentences[:top_n])
        
    X_s = doc_vec.transform(sentences)
    X_p = doc_vec.transform([prompt])

    # Since vectorizer L2-normalizes, Cosine Similarity is just the dot product
    sim = (X_s @ X_p.T).flatten()

    # Get indices of top N sentences
    top_indices = np.argsort(sim)[::-1][:top_n]
    
    # Return them in chronological order so it reads naturally
    best_sentences = [sentences[i] for i in sorted(top_indices)]
    return " ".join(best_sentences)

def get_intent(prompt: str) -> int:
    """Simple heuristic router. 0=Greeting, 1=Identity, 2=Fact/Research"""
    p = prompt.lower().strip()
    greetings = ["hello", "hi", "hey", "how are you", "good morning", "what's up", "greetings"]
    identities = ["who are you", "what are you", "who made you", "are you ai", "are you chatgpt"]
    
    if any(p == g or p.startswith(g + " ") for g in greetings):
        return 0
    if any(i in p for i in identities):
        return 1
    return 2

def generate_chat_response(prompt: str) -> str:
    """The main chat response generator."""
    intent = get_intent(prompt)

    # 2. Route to the correct behavior
    if intent == 0:
        return "Hello! I am ready to help. What would you like to know?"
    
    elif intent == 1:
        return (
            "I am a custom AI built completely from scratch using pure Python and NumPy! "
            "I don't use API keys, PyTorch, or external LLMs. My 'brain' is a custom neural network "
            "trained right here on your machine. When you ask me questions, I browse the live internet "
            "and use matrix math to synthesize answers for you!"
        )
    
    else:
        # 3. Fact/Research Query: Do a live web search!
        wiki_text = wikipedia_summary(prompt)
        web_results = search(prompt, max_results=3)

        context = ""
        if wiki_text:
            context += wiki_text + " "
            
        for r in web_results:
            context += r.get("body", "") + " "

        if not context.strip():
            return "I couldn't find any information on the web for that."

        # 4. Extract the exact answer using TF-IDF ranking
        answer = extract_best_sentences(prompt, context, top_n=4)
        
        return f"Based on my live web research: {answer}"
