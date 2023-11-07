import streamlit as st
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from features.chatbot.data.controller.ChatBotController import ChatBotController
from features.chatbot.data.datasource.Llama2DataSource import Llama2DataSource
import transformers

import gc

from features.chatbot.domain.usecase.InteractiveChat import InteractiveChat

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

def chat(input_text):
    st.info("the output of the llm")


#sider
if st.sidebar.button('Initialized 8bit'):
    ic = initialiaze_model("8bit")
if st.sidebar.button('Initialized 4bit'):
    ic = initialiaze_model("4bit")

#main
col1, col2 = st.columns(2)

with col1.form('chat'):
    text = st.text_area('Ask me something:', 'give me a list of animals?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        ic.chat(text)
"""
with col2.form('chat_rag'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        chat(text)
"""

