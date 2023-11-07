import streamlit as st
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from features.chatbot.data.controller.ChatBotController import ChatBotController
from features.chatbot.data.datasource.Llama2DataSource import Llama2DataSource
from features.chatbot.domain.usecase.InteractiveChat import InteractiveChat
import transformers
import uuid
import datetime
import gc

#create session state for rows
if "rows" not in st.session_state:
    st.session_state["rows"] = []

rows_collection = []

st.set_page_config(layout="wide")
st.title('📱Chatbot')

def initialiaze_model(mode: str):
    ic = None #making none to clean from Gpu
    gc.collect()
    if mode == "8bit":
        bnb_config = BitsAndBytesConfig(transformers.BitsAndBytesConfig(
        load_in_8bit=True))
        st.info("initializing in 8bit")
    elif mode == "4bit":
        bnb_config = BitsAndBytesConfig()
        st.info("initializing model in 4 bits")
    
    llm_source = Llama2DataSource(bnb_config)
    cb_ctr = ChatBotController([llm_source])
    ic = InteractiveChat(cb_ctr)
    return ic

def initialiaze_model_test(mode: str):
    if mode == "8bit":
        st.info("initializing in 8bit")
    elif mode == "4bit":
        st.info("initializing in 4bit")
    return "hi"


def add_row(content):
    st.session_state["rows"].append(content)

def remove_row(content):
    st.session_state["rows"].remove(content)

def generate_row(content):
    with chat_content:
        chat_content.write(content)
    return content

#sider
if st.sidebar.button('Initialized 8bit'):
    ic = initialiaze_model_test("8bit")
else:
    st.sidebar.button('Initialized 4bit')
    ic = initialiaze_model_test("4bit")

def chat(input_text):
    #st.info("asking llm")
    ic.ask_me_something(input_text)
    #add_row(f"hi {datetime.datetime.now()}")
    #st.info("the output of the llm")

#main
col1_chat, col2_chat_rag = st.columns(2)



with col1_chat.form('chat'):
    text = st.text_area('Ask me something:', 'give me a list of animals?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        chat(text)

with col2_chat_rag.form('chat_rag'):
    text = st.text_area('Ask me something:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        chat(text)

chat_content, chat_rag_content = st.columns(2)

for data in st.session_state["rows"][::-1]:
    row_data = generate_row(data)
    rows_collection.append(row_data)
