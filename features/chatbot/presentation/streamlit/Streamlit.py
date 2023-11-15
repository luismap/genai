import streamlit as st
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from features.chatbot.data.controller.ChatBotController import ChatBotController
from features.chatbot.data.datasource.FaissLangchainVectorDbSource import FaissLangchainVectorDbSource
from features.chatbot.data.datasource.Llama2DataSource import Llama2DataSource
from features.chatbot.data.models.ChatBotModel import ChatBotResponseModel
from features.chatbot.domain.usecase.InteractiveChat import InteractiveChat
import gc
import torch
import transformers
import datetime
import time
import pandas as pd

#create session state for rows
if "rows" not in st.session_state:
    st.session_state["rows"] = []
if "rows_rag" not in st.session_state:
    st.session_state["rows_rag"] = []
if "ic" not in st.session_state:
    st.session_state.ic: InteractiveChat = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "cb_ctr" not in st.session_state:
    st.session_state.cb_ctr = None
if 'startup' not in st.session_state:
    st.session_state.startup = True

rows_collection_chat = []
rows_collection_chat_rag = []

#page configs
st.set_page_config(layout="wide")
st.title('📱Gen - AI © PS labs')

#initialized widgets
qa_chat, qa_rag, vector_tab, audio_tab = st.tabs(["📝 QA", "🎓 QA rag" , "🧮 vectors", "🔈 audio"])
col1_chat = qa_chat.columns(1)[0]
col2_chat_rag = qa_rag.columns(1)[0]

#functional scripts
def initialiaze_model(mode: str):
    if st.session_state.llm is not None:
        del st.session_state.ic
        del st.session_state.cb_ctr
        del st.session_state.llm

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if mode == "8bit":
        st.session_state.startup = False
        infos = [ i.info("initializing in 8 bit") for i in [qa_rag,qa_chat]]
        #info = qa_chat.info("initializing in 8bit")
        bnb_config = BitsAndBytesConfig(transformers.BitsAndBytesConfig(
        load_in_8bit=True))
        for i in infos:
            i.empty()

    elif mode == "4bit":
        st.session_state.startup = False
        infos = [ i.info("initializing in 4 bit") for i in [qa_rag,qa_chat]]
        bnb_config = BitsAndBytesConfig()
        for i in infos:
            i.empty()

    infos = [ i.info("initializing vector store") for i in [qa_rag,qa_chat]]
    vector_db = FaissLangchainVectorDbSource()
    for i in infos:
            i.empty()

    infos = [ i.info("initializing llm") for i in [qa_rag,qa_chat]]
    llm_source = Llama2DataSource(vector_db,bnb_config)
    for i in infos:
            i.empty()

    cb_ctr = ChatBotController([llm_source])
    ic = InteractiveChat(cb_ctr)

    infos = [ i.info("model initialized") for i in [qa_rag,qa_chat]]
    time.sleep(3)
    for i in infos:
            i.empty()

    return (ic, llm_source, cb_ctr)

def initialiaze_model_test(mode: str):
    if mode == "8bit":
        st.info("initializing in 8bit")
    elif mode == "4bit":
        st.info("initializing in 4bit")
    return "hi"

def add_row(content,key):
    st.session_state[key].append(content)

def remove_row(content,key):
    st.session_state[key].remove(content)

def generate_row_chat(content, widget):
    with widget:
        widget.write(content)
    return content


def chat(input_text):
    with st.spinner("asking llm"):
        data: ChatBotResponseModel = st.session_state.ic.ask_me_something(input_text)

    model_use = data.model_use
    add_row(data.answer,"rows")
    add_row(f"`{datetime.datetime.now()}` - llm model: `{model_use}`", "rows")

    info = st.info("response generated")
    time.sleep(1)
    info.empty()

def chat_rag(input_text):
    with st.spinner("asking llm"):
        data: ChatBotResponseModel = st.session_state.ic.ask_with_rag(input_text)

    model_use = data.model_use
    add_row(data.answer,"rows_rag")
    add_row(f"`{datetime.datetime.now()}` - llm model: `{model_use}`", "rows_rag")

    info = st.info("response generated")
    time.sleep(2)
    info.empty()

def generate_web_button(urls):
    with st.spinner("generating content from the passed urls"):
        st.session_state.ic.load_from_web(urls)
    
    info = vector_tab.info("content added")
    #time.sleep(2)
    info.empty()

def vector_load_from_web(content: str):
    urls = content.strip()
    urls_parsed = urls.strip(",")
    urls_list = urls_parsed.split(",")
    vector_tab.warning(f"Following links parsed",icon="ℹ️")
    data = pd.DataFrame(urls_list,columns=["url"])
    vector_tab.dataframe(data, 
                          column_config= {"url": st.column_config.LinkColumn(" 🔗 Links Parsed")}
                          ,hide_index=True)
    
    vector_tab.button("Send", type="primary" ,on_click=generate_web_button,args=(urls_list,))
    vector_tab.button("Cancel", type="secondary")

#main
#sider
init_8bit_button = st.sidebar.button('Initialized 8bit')
init_4bit_button = st.sidebar.button('Initialized 4bit')
get_history = st.sidebar.button('show memory')

if st.session_state.startup:
    ic, llm, cb_ctr = initialiaze_model("4bit")
    st.session_state.ic = ic
    st.session_state.llm = llm
    st.session_state.cb_ctr = cb_ctr



if init_8bit_button:
    ic, llm, cb_ctr = initialiaze_model("8bit")
    st.session_state.ic = ic
    st.session_state.llm = llm
    st.session_state.cb_ctr = cb_ctr

elif init_4bit_button:
    ic, llm, cb_ctr = initialiaze_model("4bit")
    st.session_state.ic = ic
    st.session_state.llm = llm
    st.session_state.cb_ctr = cb_ctr

def get_memory():
    return st.session_state.llm._chat_chain.memory

if get_history:
    add_row(get_memory(),"rows")



#qa tab section
with col1_chat.form('chat'):
        text = st.text_area('Ask me something:', 'give me a list of animals?')
        submitted = st.form_submit_button('Submit')
        if submitted:
            chat(text)
chat_content = qa_chat.columns(1)[0]

for data in st.session_state["rows"][::-1]:
    row_data = generate_row_chat(data, chat_content)
    rows_collection_chat.append(row_data)

#qa rag tab section
with col2_chat_rag.form('chat_rag'):
        text = st.text_area('RAG - Ask me something:', 'What is cloudera cml')
        submitted = st.form_submit_button('Submit')
        if submitted:
            chat_rag(text)

chat_rag_content = qa_rag.columns(1)[0]

for data in st.session_state["rows_rag"][::-1]:
    row_data = generate_row_chat(data, chat_rag_content)
    rows_collection_chat_rag.append(row_data)

#vectors section
col_url, col_upload = vector_tab.columns(2) 


with col_url.form("url-form"):
    urls = "https://docs.cloudera.com/machine-learning/cloud/index.html,https://docs.cloudera.com/machine-learning/cloud/product/topics/ml-product-overview.html#cdsw_overview"
    text = st.text_area('Pass a list of comma separated urls:', urls)
    submitted = st.form_submit_button('Submit')
    if submitted:
        vector_load_from_web(text)

uploader = col_upload.file_uploader("Choose a CSV file", accept_multiple_files=True)

for uploaded_file in uploader:
    filename = uploaded_file.name
    bytes_data = uploaded_file.read()
    

#css
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)