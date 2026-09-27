import streamlit as st
import time
from research.chat_engine import generate_chat_response

# -- UI Configuration --
st.set_page_config(page_title="Lumina AI", page_icon="✨", layout="centered")

# Custom CSS to make it look cleaner and more like ChatGPT
st.markdown("""
<style>
    /* Hide Streamlit header and footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    /* Clean up the top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -- Sidebar (ChatGPT style) --
with st.sidebar:
    st.title("✨ Lumina AI")
    st.markdown("Your custom-built, pure-Python AI assistant.")
    
    st.markdown("---")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "🧠 **How it works:**\n"
        "This AI is built completely from scratch using NumPy. "
        "It uses a live TF-IDF vectorizer and heuristic engine to parse your intent, "
        "search the live internet, and synthesize answers using matrix math."
    )

# -- Main Chat Interface --
st.title("Lumina")

# Initialize chat history
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input Box
if prompt := st.chat_input("Message Lumina..."):
    
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Show assistant thinking & response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*(Thinking...)*")

        try:
            full_response = generate_chat_response(prompt)

            # Simulated streaming "typing" effect
            streamed_text = ""
            for word in full_response.split(" "):
                streamed_text += word + " "
                message_placeholder.markdown(streamed_text + "▌")
                time.sleep(0.04)  # Typing speed
            
            # Final text without the cursor
            message_placeholder.markdown(streamed_text)
            
        except Exception as e:
            full_response = f"Oops! I encountered an error while searching: {e}"
            message_placeholder.markdown(full_response)

    # 3. Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
