import streamlit as st
import datetime

#create session state for rows
if "rows" not in st.session_state:
    st.session_state["rows"] = []

if "ic" not in st.session_state:
    st.session_state.ic = None

rows_collection = []

st.set_page_config(layout="wide")
st.title('📱Chatbot')

def add_row(content):
    st.session_state["rows"].append(content)

def remove_row(content):
    st.session_state["rows"].remove(content)

def generate_row(content):
    with chat_content:
        chat_content.write(content)
    return content

#sider



def chat(input_text):
    info = st.info("asking llm")
    add_row(st.session_state.ic.ask_me_something(input_text).answer)
    info.empty()
    add_row(f"################### {datetime.datetime.now()}")
    info = st.info("response generated")
    info.empty()

#main
col1_chat, col2_chat_rag = st.columns(2)

with col1_chat.form('chat'):
    text = st.text_area('Ask me something:', 'give me a list of animals?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        chat(text)

chat_content = st.columns(1)

for data in st.session_state["rows"][::-1]:
    row_data = generate_row(data)
    rows_collection.append(row_data)