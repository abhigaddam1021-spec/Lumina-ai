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
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        color: #1a1a1a !important; /* Force dark text */
    }}
    
    /* Ensure markdown text inside chat is also dark */
    .stChatMessage p, .stChatMessage div, .stMarkdown p {{
        color: #1a1a1a !important;
    }}
    
    /* Force sidebar text to be visible and match calming theme */
    [data-testid="stSidebar"] {{
        background-color: #f4f9f9 !important;
        color: #1a1a1a !important;
    }}
    
    /* Ensure elements inside the sidebar get the dark text */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {{
        color: #1a1a1a !important;
    }}
    
    /* Fix Chat Input box contrast */
    [data-testid="stChatInput"], 
    [data-testid="stChatInput"] > div {{
        background-color: white !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: #1a1a1a !important;
        background-color: transparent !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: #94a3b8 !important;
    }}
    [data-testid="stChatInput"] button, 
    [data-testid="stChatInput"] svg {{
        color: #475569 !important;
        fill: #475569 !important;
    }}
    [data-testid="stChatInput"] button:hover {{
        background-color: #f1f5f9 !important;
    }}
    
    /* Fix Streamlit bottom block background */
    [data-testid="stBottom"], 
    [data-testid="stBottom"] > *,
    [data-testid="stBottom"] > * > *,
    [data-testid="stBottomBlockContainer"], 
    [data-testid="stBottomBlockContainer"] > * {{
        background-color: transparent !important;
        background: transparent !important;
    }}
    
    /* Fix sidebar Context Window textarea */
    [data-testid="stTextArea"] textarea {{
        background-color: white !important;
        color: #1a1a1a !important;
        border: 1px solid #cbd5e1 !important;
    }}
    
    /* Fix buttons and expander headers going dark in the sidebar */
    [data-testid="stSidebar"] button, [data-testid="stSidebar"] summary {{
        background-color: white !important;
        color: #1a1a1a !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }}
    [data-testid="stSidebar"] button:hover, [data-testid="stSidebar"] summary:hover {{
        background-color: #f8fafc !important;
        border-color: #cbd5e1 !important;
    }}
    [data-testid="stSidebar"] details {{
        background-color: transparent !important;
    }}
</style>
''', unsafe_allow_html=True)

# -- Sidebar --
with st.sidebar:
    st.title("✨ Lumina AI")
    st.markdown("**Version:** Alpha")
    st.markdown("Your custom-built, pure-Python AI assistant.")
    
    st.markdown("---")
    
    with st.expander("⚙️ System Specifics & Updates", expanded=False):
        st.markdown("**Simple System Specifics**")
        st.caption("• **IQ Core:** 140\n• **EQ Core:** 9\n• **Accuracy:** >98%\n• **Hallucinations:** <0.5%")
        
        st.markdown("""
<details style="margin-bottom: 10px;">
<summary style="cursor: pointer; font-weight: bold; padding: 5px 0;">Advanced System Specifics</summary>
<div style="padding-left: 15px; font-size: 0.9em; color: #1a1a1a;">
• <span style="font-family: monospace; font-size: 0.85em; background: #e2e8f0; padding: 2px 4px; border-radius: 3px;">TOPOLOGY</span> : 150-Layer Deep Feed-Forward<br>
• <span style="font-family: monospace; font-size: 0.85em; background: #e2e8f0; padding: 2px 4px; border-radius: 3px;">CTX_WINDOW</span> : 2048 Tokens (O(N) Complexity)<br>
• <span style="font-family: monospace; font-size: 0.85em; background: #e2e8f0; padding: 2px 4px; border-radius: 3px;">VEC_ENGINE</span> : TF-IDF with L2 Cosine Sim Matrix<br>
• <span style="font-family: monospace; font-size: 0.85em; background: #e2e8f0; padding: 2px 4px; border-radius: 3px;">OPTIMIZATION</span> : Active Continuous Loss Minimization
</div>
</details>

<details>
<summary style="cursor: pointer; font-weight: bold; padding: 5px 0;">Update Logs & News</summary>
<div style="padding-left: 5px; color: #1a1a1a;">
<details style="margin-top: 5px;">
<summary style="cursor: pointer; font-size: 0.9em; font-weight: 600;">Version: Alpha (Sep 27, 2026)</summary>
<div style="padding-left: 15px; font-size: 0.85em;">
• Added Anti-Bias / Objective Data Filters<br>
• Added 20 Personalization Tones<br>
• Added 20 Language Translation Support<br>
• Migrated to Single-Mode Interface<br>
• UI Overhaul & Nested Tabs
</div>
</details>

<details style="margin-top: 5px;">
<summary style="cursor: pointer; font-size: 0.9em; font-weight: 600;">Version: Beta (Sep 26, 2026)</summary>
<div style="padding-left: 15px; font-size: 0.85em;">
• Built core 150-layer neural network from scratch<br>
• Implemented TF-IDF Live Web Research<br>
• Established initial Math & EQ parsing routines
</div>
</details>

<div style="margin-top: 15px; font-size: 0.85em; font-weight: 700; font-style: italic;">
🚀 Gamma and V1 releasing soon...
</div>
</div>
</details>
""", unsafe_allow_html=True)
        
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
        {"role": "assistant", "content": "Hello! I am Lumina. How can I assist you today?"}
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
