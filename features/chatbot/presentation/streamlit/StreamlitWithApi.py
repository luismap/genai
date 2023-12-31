import streamlit as st
import datetime
import requests
import time
import os

#create session state for rows
if "rows" not in st.session_state:
    st.session_state["rows"] = []

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
url = os.getenv("url")
bearer = f'Bearer  {os.getenv("bearer")}'

def post_llm_api(question:str) -> dict:
    data = '{"request":{"question":"' + question +  '"}}'
    r = requests.post(url, data=data, headers={'Content-Type': 'application/json', 'Authorization': bearer})
    return r.json()

def chat(input_text):
    with st.spinner("asking llm"):
        data = post_llm_api(input_text)

    model_use = data["response"]["model_use"]
    add_row(data["response"]["answer"])
    add_row(f"{datetime.datetime.now()} - llm model: {model_use}")
    
    info = st.info("response generated")
    time.sleep(1)
    info.empty()

#main
col1_chat = st.columns(1)[0]

with col1_chat.form('chat'):
    text = st.text_area('Ask me something:', 'give me a list of animals?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        chat(text)

chat_content = st.columns(2)[0]

for data in st.session_state["rows"][::-1]:
    row_data = generate_row(data)
    rows_collection.append(row_data)