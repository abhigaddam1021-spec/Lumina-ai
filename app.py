import streamlit as st
import time
from research.chat_engine import generate_chat_response

# -- UI Configuration --
st.set_page_config(page_title="Lumina AI", page_icon="✨", layout="centered")

# Default values for personalization
if "font_style" not in st.session_state: st.session_state.font_style = "System Default"
if "tone" not in st.session_state: st.session_state.tone = "Professional"
if "language" not in st.session_state: st.session_state.language = "English"

font_map = {
    "System Default": "system-ui, -apple-system, sans-serif",
    "Serif": "Georgia, serif",
    "Monospace": "'Courier New', Courier, monospace",
    "Comic Sans MS": "'Comic Sans MS', cursive",
    "Impact": "Impact, fantasy",
    "Verdana": "Verdana, sans-serif"
}

chosen_font = font_map.get(st.session_state.font_style, "system-ui")

# Custom CSS for a calming, relaxed vibe
st.markdown(f'''
<style>
    /* Calming color palette and Font */
    html, body, [class*="css"] {{
        font-family: {chosen_font} !important;
    }}
    .stApp {{
        background: linear-gradient(to bottom right, #f4f9f9, #e0f0f5);
        color: #2c3e50;
    }}
    
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
    
    /* Chat message styling for softer look */
    .stChatMessage {{
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }}
</style>
''', unsafe_allow_html=True)

# -- Sidebar --
with st.sidebar:
    st.title("✨ Lumina AI")
    st.markdown("**Version:** IQ 140 | EQ 9")
    st.markdown("Your custom-built, pure-Python AI assistant.")
    
    st.markdown("---")
    
    with st.expander("🎨 Personalisation", expanded=False):
        tones = [
            "Professional", "Empathetic", "Humorous", "Sarcastic", "Poetic", 
            "Academic", "Casual", "Enthusiastic", "Pirate", "Robot", 
            "Philosophical", "Mystical", "Direct", "Gentle", "Optimistic", 
            "Pessimistic", "Dramatic", "Stoic", "Whimsical", "Sassy"
        ]
        langs = [
            "English", "Spanish", "French", "German", "Chinese (Simplified)", 
            "Japanese", "Korean", "Hindi", "Arabic", "Russian", 
            "Portuguese", "Italian", "Dutch", "Turkish", "Polish", 
            "Swedish", "Indonesian", "Vietnamese", "Thai", "Greek"
        ]
        fonts = list(font_map.keys())
        
        st.session_state.tone = st.selectbox("Tone", tones, index=tones.index(st.session_state.tone))
        st.session_state.language = st.selectbox("Language", langs, index=langs.index(st.session_state.language))
        st.session_state.font_style = st.selectbox("Font Style", fonts, index=fonts.index(st.session_state.font_style))

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Large Text Analysis**")
    st.caption("Paste 1000+ lines of code or text here, then ask questions about it in the chat.")
    large_text = st.text_area("Context Window", height=150, placeholder="Paste huge text blocks here...")
    
    st.markdown("---")
    st.caption("⚠️ **Training Status:** Continuous Deep-Layer Training is currently ACTIVE in the background. Do not close unless requested.")

# -- Main Chat Interface --
st.title("Lumina")

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am feeling very relaxed today. How can I help you?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message Lumina..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*(Processing...)*")

        try:
            full_response = generate_chat_response(
                prompt, 
                large_context=large_text,
                tone=st.session_state.tone,
                language=st.session_state.language
            )

            streamed_text = ""
            for word in full_response.split(" "):
                streamed_text += word + " "
                message_placeholder.markdown(streamed_text + "▌")
                time.sleep(0.04) 
            
            message_placeholder.markdown(streamed_text)
            
        except Exception as e:
            full_response = f"Oops! System Error: {e}"
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
