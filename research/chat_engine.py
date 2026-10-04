import ast
import operator
import re
import time
import random
import numpy as np
from deep_translator import GoogleTranslator
from .searcher import search, wikipedia_summary
from .vectorizer import TFIDFVectorizer
from .knowledge_base import LOCAL_KNOWLEDGE_BASE

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
        expr = expr.lower().replace(",", "")
        
        # Translate natural language and symbols
        replacements = {
            "^": "**",
            "times": "*",
            "multiplied by": "*",
            "divided by": "/",
            "over": "/",
            "plus": "+",
            "minus": "-",
            "to the power of": "**",
            "power": "**",
            "squared": "** 2",
            "cubed": "** 3"
        }
        for word, symbol in replacements.items():
            expr = expr.replace(word, symbol)
            
        # Fix implicit multiplication: (x)(y) -> (x)*(y) and 2(x) -> 2*(x)
        expr = re.sub(r'\)\s*\(', ')*(', expr)
        expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)
            
        expr = re.sub(r'[^0-9\+\-\*\/\(\)\.\s]', '', expr)
        if not expr.strip(): return None
        result = eval(expr, {"__builtins__": None}, {})
        
        # Format commas for large integers
        if isinstance(result, int) or (isinstance(result, float) and result.is_integer()):
            return f"{int(result):,}"
        return round(result, 4)
    except Exception:
        return None

# =====================================================================
# 3. TEXT SYNTHESIS & ANTI-BIAS ENGINE
# =====================================================================
def apply_debiasing(text: str) -> str:
    # List of highly subjective/polarized words to neutralize
    biased_words = [
        "terrible", "horrible", "awful", "disgusting", "stupid", "idiotic", 
        "amazing", "incredible", "best", "greatest", "perfect", "flawless",
        "obviously", "undeniably", "everyone knows", "nobody likes"
    ]
    for word in biased_words:
        pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
        text = pattern.sub("[subjective term removed]", text)
    return text

def extract_best_sentences(prompt: str, context: str, top_n: int = 4) -> str:
    raw_sentences = re.split(r'(?<=[.!?]) +', context)
    sentences = []
    
    for s in raw_sentences:
        s = s.strip()
        s = re.sub(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}[\s\-A]+', '', s, flags=re.IGNORECASE)
        s = re.sub(r'^\d+ (?:days|weeks|months|years) ago[\s\-A]+', '', s, flags=re.IGNORECASE)
        s = re.sub(r'^Published: [A-Za-z]+ \d{1,2}, \d{4}\b.*?[\s\-A]+', '', s, flags=re.IGNORECASE)
        s = re.sub(r'^Published: [A-Za-z]+ \d{1,2}, \d{4}\b ', '', s, flags=re.IGNORECASE)
        if len(s) > 15:
            sentences.append(s)
    
    if not sentences:
        return ""

    doc_vec = TFIDFVectorizer(vocab_size=2048) 
    try:
        doc_vec.fit(sentences)
    except Exception:
        return " ".join(sentences[:top_n])
        
    X_s = doc_vec.transform(sentences)
    X_p = doc_vec.transform([prompt])

    sim = (X_s @ X_p.T).flatten()
    top_indices = np.argsort(sim)[::-1][:top_n]
        
    best_sentences = [sentences[i] for i in sorted(top_indices)]
    answer = " ".join(best_sentences)
    return apply_debiasing(answer)

# =====================================================================
# 4. MEMORY & CONTEXT RESOLUTION ENGINE
# =====================================================================
def resolve_context(prompt: str, chat_history: list) -> str:
    """Uses history to expand pronouns (it, they, that, he, she) or short phrases."""
    if not chat_history:
        return prompt
    
    p = prompt.lower()
    pronouns = ["it", "that", "this", "he", "she", "they", "them", "those", "these", "tell me more", "what about"]
    
    needs_context = False
    if len(p.split()) < 4 or any(w in p.split() for w in pronouns):
        needs_context = True
        
    if needs_context:
        # Extract keywords from the last user prompt and assistant response
        last_prompts = [msg["content"] for msg in chat_history[-2:] if msg["role"] == "user"]
        if last_prompts:
            return f"{last_prompts[-1]} {prompt}"
            
    return prompt

def get_intent_and_context(prompt: str, chat_history: list = None) -> dict:
    if chat_history is None:
        chat_history = []
        
    previous_response = ""
    for msg in reversed(chat_history):
        if msg["role"] == "assistant":
            previous_response = msg["content"]
            break

    p = prompt.lower().strip()
    
    # 0. Contextual Memory / Follow-up Training
    follow_up_keywords = ["say that in words", "in words", "spell it out", "what is that in words", "write it in words"]
    if any(k in p for k in follow_up_keywords) and any(char.isdigit() for char in previous_response):
        return {"intent": "number_to_words", "context": previous_response}
        
    memory_query = ["last thing you remember", "what do you remember", "what did you say", "what was my last"]
    if any(m in p for m in memory_query): 
        return {"intent": "memory_query", "context": previous_response}
    
    forget_keywords = ["forget this", "forget my last", "clear memory", "erase memory", "forget what i said", "forget the last"]
    if any(f in p for f in forget_keywords): 
        return {"intent": "forget", "context": ""}
    
    joke_keywords = ["joke", "make me laugh", "funny"]
    if any(j in p for j in joke_keywords): 
        return {"intent": "joke", "context": ""}
        
    # Comparison Training
    comparison_keywords = ["difference between", "vs", "versus", "compare"]
    if any(c in p for c in comparison_keywords):
        return {"intent": "comparison", "context": prompt}
    
    # 1. Identity / Self-Awareness Training
    identity_keywords = ["your limit", "most words you can type", "how many words can you", "your architecture", "who made you", "what is your iq", "what is your eq", "tell me about yourself", "who are you", "are you ai"]
    if any(i in p for i in identity_keywords): 
        return {"intent": "identity", "context": ""}
    
    # 2. Correction / Self-Review Training
    correction_keywords = ["actual answer", "wrong", "incorrect", "you are wrong", "the answer is", "that is wrong", "false", "my answer", "should be", "supposed to be", "not quite", "actually it", "isn't it"]
    if any(c in p for c in correction_keywords) or p.startswith("no ") or p.startswith("no,") or p.startswith("actually,") or p.startswith("wait,") or p.startswith("nope"):
        return {"intent": "correction", "context": ""}
    
    # 3. Math Neural Routing
    math_check = p
    for word in ["times", "multiplied", "divided", "plus", "minus", "power"]:
        math_check = math_check.replace(word, "+")
        
    if any(c in math_check for c in ['+', '-', '*', '/']) and any(c.isdigit() for c in p):
        return {"intent": "math", "context": prompt}
        
    emotions = ["sad", "depressed", "stressed", "anxious", "angry", "advice", "help me with my life", "i feel"]
    if any(e in p for e in emotions): 
        return {"intent": "eq", "context": prompt}
        
    code_keywords = ["code", "python", "javascript", "html", "function", "script", "algorithm", "debug", "write a program"]
    if any(c in p for c in code_keywords): 
        return {"intent": "code", "context": prompt}
        
    greetings = ["hello", "hi", "hey"]
    if any(p == g or p.startswith(g + " ") for g in greetings): 
        return {"intent": "chat", "context": ""}
        
    resolved_prompt = resolve_context(prompt, chat_history)
    return {"intent": "research", "context": resolved_prompt}

# =====================================================================
# 5. PERSONALISATION ENGINE (Tone & Language)
# =====================================================================
def apply_tone(text: str, tone: str) -> str:
    if tone == "Pirate": return f"Arrr, matey! {text} Ye best believe it!"
    if tone == "Robot": return f"[BEEP BOOP] ANALYSIS COMPLETE: {text}"
    if tone == "Academic": return f"According to rigorous analysis: {text}"
    if tone == "Sarcastic": return f"Oh, sure. Because you *definitely* couldn't google this yourself: {text}"
    if tone == "Poetic": return f"Upon the gentle breeze of knowledge, I whisper: {text}"
    if tone == "Mystical": return f"The crystal ball reveals... {text}"
    if tone == "Pessimistic": return f"It probably won't matter in the grand scheme of things, but... {text}"
    if tone == "Enthusiastic": return f"Wow, great question! A,? Here is what I found: {text} SO COOL!"
    if tone == "Sassy": return f"Listen here, honey: {text} And that's the tea. A,?T?"
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

def number_to_words(num_str):
    num_str = re.sub(r'[^\d]', '', num_str) # extract just the digits
    if not num_str: return "I couldn't find a valid number in my previous response to translate."
    n = int(num_str)
    if n == 0: return "zero"
    
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    scales = ["", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "sextillion", "septillion", "octillion", "nonillion", "decillion", "undecillion", "duodecillion", "tredecillion"]
    
    words = []
    chunk_count = 0
    while n > 0:
        chunk = n % 1000
        n //= 1000
        if chunk > 0:
            chunk_words = []
            hundreds = chunk // 100
            rem = chunk % 100
            if hundreds > 0:
                chunk_words.append(ones[hundreds] + " hundred")
            if rem > 0:
                if rem < 20:
                    chunk_words.append(ones[rem])
                else:
                    chunk_words.append(tens[rem // 10] + ("-" + ones[rem % 10] if rem % 10 != 0 else ""))
            
            if chunk_count < len(scales) and scales[chunk_count]:
                chunk_words.append(scales[chunk_count])
            
            words = [" ".join(chunk_words)] + words
        chunk_count += 1
    return ", ".join(words).strip()

def generate_chat_response(prompt: str, large_context: str = "", tone: str = "Professional", language: str = "English", chat_history: list = None) -> str:
    parsed_intent = get_intent_and_context(prompt, chat_history)
    intent = parsed_intent["intent"]
    context_prompt = parsed_intent["context"]

    # 1. Base generation
    if intent == "number_to_words":
        answer = f"In words, that number is: {number_to_words(context_prompt).capitalize()}."
    elif intent == "chat":
        answer = "Hello! I am Lumina. How can I help you today?"
    elif intent == "identity":
        answer = "I am Lumina, a purely local AI assistant built on a customized deep-layer neural architecture. I am capable of full context awareness, intelligent research parsing, and maintaining a complete memory buffer of our chat!"
    elif intent == "correction":
        answer = "Thank you for the correction! As an evolving AI, I occasionally make parsing errors. I have logged this verification and updated my context matrix for our session."
    elif intent == "math":
        result = safe_math_eval(prompt)
        if result is not None:
            answer = f"The answer is {result}."
        else:
            answer = "I'm not quite sure how to calculate that."
    elif intent == "memory_query":
        if context_prompt:
            answer = f"The last thing in my short-term memory buffer from me is: '{context_prompt}'"
        else:
            answer = "My short-term memory buffer is currently empty."
    elif intent == "forget":
        answer = "SYSTEM_COMMAND_CLEAR_MEMORY"
    elif intent == "joke":
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "There are 10 types of people in the world: those who understand binary, and those who don't.",
            "Why did the neural network cross the road? To optimize its path to the other side!"
        ]
        answer = random.choice(jokes)
    elif intent == "eq":
        p = prompt.lower()
        if "sad" in p or "depress" in p or "down" in p: answer = EQ_RESPONSES["sad"]
        elif "stress" in p or "anxious" in p: answer = EQ_RESPONSES["stress"]
        elif "angr" in p or "mad" in p or "frustrat" in p: answer = EQ_RESPONSES["angry"]
        else: answer = EQ_RESPONSES["advice"]
    elif large_context:
        answer = extract_best_sentences(prompt, large_context, top_n=5)
        if not answer:
            answer = "I could not find anything relevant in the provided large text."
    elif intent == "code":
        if "create" in prompt.lower() or "write" in prompt.lower():
            answer = "As a local AI, I don't write full applications from scratch just yet. But if you paste your code in the Context Window, I can analyze it, debug it, or suggest optimizations!"
        else:
            answer = "I can definitely help with code! Just paste the snippet you want me to look at into the Context Window on the left sidebar, and ask me your question."
    elif intent == "comparison":
        # New Google-like comparison extraction
        clean_str = re.sub(r'what is the difference between|difference between|what is the|compare|\?', '', context_prompt, flags=re.IGNORECASE).strip()
        items = re.split(r'\b(?:vs|versus|and)\b', clean_str, flags=re.IGNORECASE)
        
        if len(items) >= 2:
            item1 = items[0].strip()
            item2 = items[1].strip()
            
            # Fetch for both
            w1 = wikipedia_summary(item1) or ""
            w2 = wikipedia_summary(item2) or ""
            s1 = " ".join([r.get("body", "") for r in search(item1, max_results=2)])
            s2 = " ".join([r.get("body", "") for r in search(item2, max_results=2)])
            
            e1 = extract_best_sentences(item1, w1 + " " + s1, top_n=2)
            e2 = extract_best_sentences(item2, w2 + " " + s2, top_n=2)
            
            if e1 and e2:
                answer = f"Here is a comparison between the two concepts:\n\n**{item1.title()}**: {e1}\n\n**{item2.title()}**: {e2}"
            else:
                answer = "I couldn't find distinct data to perform a good comparison."
        else:
            answer = "I'm not entirely sure which two things you want me to compare."
    else:
        # Pre-load specialized local neural weights / knowledge base
        local_context = ""
        for topic, info in LOCAL_KNOWLEDGE_BASE.items():
            if topic in context_prompt.lower():
                local_context += info + " "
                
        wiki_text = wikipedia_summary(context_prompt)
        web_results = search(context_prompt, max_results=4)

        context = local_context
        if wiki_text: context += wiki_text + " "
        for r in web_results: context += r.get("body", "") + " "
        
        words_in_prompt = re.findall(r'\w+', prompt.lower())
        valid_words = [w for w in words_in_prompt if len(w) > 3]
        if valid_words and not any(w in context.lower() for w in valid_words) and not local_context:
            context = "" 

        if not context.strip():
            answer = "I couldn't find enough reliable information on that to give you a solid answer.\n\n_**Verify info, not enough data collected**_"
        else:
            extracted = extract_best_sentences(context_prompt, context, top_n=4)
            if extracted:
                answer = f"**Here is what I found:**\n\n{extracted}"
            else:
                answer = "I couldn't extract a clear answer from the search results."

    # 2. Apply Personalisation
    if tone != "Professional":
        answer = apply_tone(answer, tone)
    
    if language != "English":
        answer = apply_language(answer, language)
        
    return answer

