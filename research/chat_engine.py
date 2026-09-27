import ast
import operator
import re
import time
import numpy as np
from deep_translator import GoogleTranslator
from .searcher import search, wikipedia_summary
from .vectorizer import TFIDFVectorizer

# =====================================================================
# 1. EMOTIONAL INTELLIGENCE (EQ) & ADVICE ENGINE
# =====================================================================
EQ_RESPONSES = {
    "sad": "I'm really sorry you're feeling this way. It's completely normal to feel down sometimes. Please remember to be kind to yourself, and consider talking to a friend or professional if it gets too heavy.",
    "stress": "It sounds like you're under a lot of pressure. Remember to take things one step at a time, breathe deeply, and give yourself permission to rest. You've got this.",
    "angry": "It is completely valid to feel frustrated right now. When you're ready, taking a step back and breathing might help clear your mind. I'm here to listen.",
    "advice": "When facing a tough decision, it often helps to write down your core values and see which option aligns best with them. Trust your intuition, but don't be afraid to ask for help."
}

# =====================================================================
# 2. MATHEMATICAL REASONING ENGINE
# =====================================================================
def safe_math_eval(expr):
    try:
        expr = re.sub(r'[^0-9\+\-\*\/\(\)\.\s]', '', expr)
        if not expr.strip(): return None
        result = eval(expr, {"__builtins__": None}, {})
        return result
    except Exception:
        return None

# =====================================================================
# 3. HIGH-ACCURACY & ANTI-BIAS TEXT EXTRACTION
# =====================================================================
def apply_debiasing(text: str) -> str:
    """Removes biased/polarized adjectives to maintain objective AI neutrality."""
    # List of highly subjective/polarized words to neutralize
    biased_words = [
        "terrible", "horrible", "awful", "disgusting", "stupid", "idiotic", 
        "amazing", "incredible", "best", "greatest", "perfect", "flawless",
        "obviously", "undeniably", "everyone knows", "nobody likes"
    ]
    # Replace biased words with neutral equivalents or strip them
    for word in biased_words:
        # Case insensitive replacement
        pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
        text = pattern.sub("[subjective term removed]", text)
    return text

def extract_best_sentences(prompt: str, context: str, top_n: int = 4) -> str:
    sentences = re.split(r'(?<=[.!?]) +', context)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    if not sentences:
        return "I couldn't find enough detailed data to form a highly accurate answer.\n\n_**Verify info, not enough data collected**_"

    doc_vec = TFIDFVectorizer(vocab_size=2048) 
    try:
        doc_vec.fit(sentences)
    except Exception:
        return " ".join(sentences[:top_n])
        
    X_s = doc_vec.transform(sentences)
    X_p = doc_vec.transform([prompt])

    sim = (X_s @ X_p.T).flatten()

    top_indices = np.argsort(sim)[::-1][:top_n]
    low_confidence = False
    
    # Check hallucination threshold
    if len(top_indices) == 0 or sim[top_indices[0]] <= 0.05:
        low_confidence = True
        
    best_sentences = [sentences[i] for i in sorted(top_indices)]
    answer = " ".join(best_sentences)
    
    # 1. Apply Debias Engine
    answer = apply_debiasing(answer)
    
    # 2. Append Hallucination Warning if similarity is too low
    if low_confidence:
        answer += "\n\n_**Verify info, not enough data collected**_"
        
    return answer

def get_intent(prompt: str) -> str:
    p = prompt.lower().strip()
    if any(char in p for char in ['+', '-', '*', '/']) and any(char.isdigit() for p in p):
        if re.search(r'\d[\+\-\*\/]\d', p.replace(" ", "")): return "math"
    emotions = ["sad", "depressed", "stressed", "anxious", "angry", "advice", "help me with my life", "i feel"]
    if any(e in p for e in emotions): return "eq"
    code_keywords = ["code", "python", "javascript", "html", "function", "script", "algorithm", "debug", "write a program"]
    if any(c in p for c in code_keywords): return "code"
    greetings = ["hello", "hi", "hey", "who are you", "are you ai"]
    if any(p == g or p.startswith(g + " ") for g in greetings): return "chat"
    return "research"

# =====================================================================
# 5. PERSONALISATION ENGINE (Tone & Language)
# =====================================================================
def apply_tone(text: str, tone: str) -> str:
    if tone == "Pirate": return f"Arrr, matey! {text} Ye best believe it!"
    if tone == "Robot": return f"[BEEP BOOP] ANALYSIS COMPLETE: {text}"
    if tone == "Academic": return f"According to rigorous, unbiased analysis of the provided dataset: {text}"
    if tone == "Sarcastic": return f"Oh, sure. Because you *definitely* couldn't google this yourself: {text}"
    if tone == "Poetic": return f"Upon the gentle breeze of knowledge, I whisper: {text}"
    if tone == "Mystical": return f"The crystal ball reveals... {text}"
    if tone == "Pessimistic": return f"It probably won't matter in the grand scheme of things, but... {text}"
    if tone == "Enthusiastic": return f"Wow, great question! ðŸŽ‰ Here is what I found: {text} SO COOL!"
    if tone == "Sassy": return f"Listen here, honey: {text} And that's the tea. ðŸ’…"
    if tone == "Stoic": return f"Accept this objective reality: {text}"
    if tone == "Humorous": return f"{text} But hey, don't quote me on that, I'm just a bunch of matrices!"
    if tone == "Gentle": return f"Take a deep breath. {text} Everything will be okay."
    if tone == "Dramatic": return f"BEHOLD! The unbiased truth you seek: {text}"
    return text

def apply_language(text: str, lang: str) -> str:
    if lang == "English": return text
    lang_codes = {
        "Spanish": "es", "French": "fr", "German": "de", "Chinese (Simplified)": "zh-CN",
        "Japanese": "ja", "Korean": "ko", "Hindi": "hi", "Arabic": "ar",
        "Russian": "ru", "Portuguese": "pt", "Italian": "it", "Dutch": "nl",
        "Turkish": "tr", "Polish": "pl", "Swedish": "sv", "Indonesian": "id",
        "Vietnamese": "vi", "Thai": "th", "Greek": "el"
    }
    code = lang_codes.get(lang)
    if not code: return text
    try:
        return GoogleTranslator(source='auto', target=code).translate(text)
    except Exception:
        return text + " [Translation Error]"

def generate_chat_response(prompt: str, large_context: str = "", tone: str = "Professional", language: str = "English") -> str:
    intent = get_intent(prompt)

    # 1. Base generation
    if intent == "chat":
        answer = "Hello! I am Lumina. How can I help you today?"
    elif intent == "math":
        result = safe_math_eval(prompt)
        if result is not None:
            answer = f"The answer is {result}."
        else:
            answer = "I'm not quite sure how to calculate that."
    elif intent == "eq":
        p = prompt.lower()
        if "sad" in p or "depress" in p or "down" in p: answer = EQ_RESPONSES["sad"]
        elif "stress" in p or "anxious" in p: answer = EQ_RESPONSES["stress"]
        elif "angr" in p or "mad" in p or "frustrat" in p: answer = EQ_RESPONSES["angry"]
        else: answer = EQ_RESPONSES["advice"]
    elif large_context:
        answer = extract_best_sentences(prompt, large_context, top_n=5)
    elif intent == "code":
        answer = "I can definitely help with code! Just paste the snippet you want me to look at into the Context Window on the left sidebar, and ask me your question."
    else:
        # Append 'objective facts' to force unbiased web search results
        wiki_text = wikipedia_summary(prompt)
        web_results = search(prompt + " objective facts neutral overview", max_results=4)

        context = ""
        if wiki_text: context += wiki_text + " "
        for r in web_results: context += r.get("body", "") + " "

        if not context.strip():
            answer = "I couldn't find enough reliable information on that to give you a solid answer.\n\n_**Verify info, not enough data collected**_"
        else:
            answer = extract_best_sentences(prompt, context, top_n=4)

    # 2. Apply Personalisation
    if tone != "Professional":
        answer = apply_tone(answer, tone)
    
    if language != "English":
        answer = apply_language(answer, language)
        
    return answer
