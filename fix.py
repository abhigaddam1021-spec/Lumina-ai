with open(r'c:\Abhiram\Antigravity\Project 1\app.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re
# The weird broken characters in app.py are actually actual literal '?' if powershell mangled them, 
# or they are something else. Let's just find and replace them directly.
text = re.sub(r'page_icon="[^"]+"', 'page_icon="\u2728"', text)
text = re.sub(r'st\.title\("[^"]* Lumina AI"\)', 'st.title("\u2728 Lumina AI")', text)
text = re.sub(r'st\.expander\("[^"]* System Specifics & Updates"', 'st.expander("\u2699\uFE0F System Specifics & Updates"', text)
text = re.sub(r'st\.expander\("[^"]* Personalisation"', 'st.expander("\U0001F3A8 Personalisation"', text)
text = re.sub(r'st\.button\("[^"]*  New Chat"', 'st.button("\u2795  New Chat"', text)

with open(r'c:\Abhiram\Antigravity\Project 1\app.py', 'w', encoding='utf-8') as f:
    f.write(text)
