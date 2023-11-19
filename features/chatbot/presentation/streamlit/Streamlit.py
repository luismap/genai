import streamlit as st
from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from features.chatbot.data.controller.ChatBotController import ChatBotController
from features.chatbot.data.controller.AudioController import AudioController
from features.chatbot.data.datasource.FaissLangchainVectorDbSource import FaissLangchainVectorDbSource
from features.chatbot.data.datasource.ChatBotLlama2DataSource import Llama2DataSource
from features.chatbot.data.datasource.WhisperDataSource import WhisperDataSource
from features.chatbot.data.models.ChatBotModel import ChatBotResponseModel
from features.chatbot.domain.usecase.InteractiveChat import InteractiveChat
from features.chatbot.domain.usecase.AudioTask import AudioTask
from features.chatbot.data.models.AudioDataModel import AudioDataReadModel
import gc
import torch
import transformers
import datetime
import time
import pandas as pd
from pathlib import Path

#create session state for rows
if "rows" not in st.session_state:
    st.session_state["rows"] = []
if "rows_rag" not in st.session_state:
    st.session_state["rows_rag"] = []
if "rows_audio" not in st.session_state:
    st.session_state["rows_audio"] = []
if "rows_audio_log" not in st.session_state:
    st.session_state["rows_audio_log"] = []
if "ic" not in st.session_state:
    st.session_state.ic: InteractiveChat = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "cb_ctr" not in st.session_state:
    st.session_state.cb_ctr = None
if 'startup' not in st.session_state:
    st.session_state.startup = True
if 'current_qbit_mode' not in st.session_state:
    st.session_state.current_qbit_mode = ""
if "audio_task" not in st.session_state:
    st.session_state.audio_task: AudioTask = None
if "audio_ctr" not in st.session_state:
    st.session_state.audio_ctr: AudioController = None
if "whisper_ds" not in st.session_state:
    st.session_state.whisper_ds: WhisperDataSource = None
if "audio_files" not in st.session_state:
    st.session_state.audio_files = []

rows_collection_chat = []
rows_collection_chat_rag = []
rows_collection_audio = []
rows_collection_audio_log = []

#page configs
st.set_page_config(layout="wide")
st.title('📱Gen - AI © PS labs')

#initialized widgets
qa_chat, qa_rag, vector_tab, audio_tab = st.tabs(["📝 QA", "🎓 QA rag" , "🧮 vectors", "🔈 audio"])
col1_chat = qa_chat.columns(1)[0]
col2_chat_rag = qa_rag.columns(1)[0]

#functional scripts

def color_text(color: str, text: str):
    text = f"<span style=\"color:{color}\">{text}</span>"
    return text

def initialize_audio():
    st.session_state.startup = False
    if st.session_state.whisper_ds is not None:
        del st.session_state.audio_task
        del st.session_state.audio_ctr
        del st.session_state.whisper_ds

    info = audio_tab.info("Initializing audio model")
    whisper_ds = WhisperDataSource()
    info.empty()
    
    info = audio_tab.info("Initializing audio controller")
    audio_ctr = AudioController([whisper_ds])
    info.empty()

    info = audio_tab.info("Initializing audio task")
    audio_task = AudioTask(audio_ctr)
    info.empty()

    return (whisper_ds, audio_ctr, audio_task)

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
        st.session_state.current_qbit_mode = "8bit"
        infos = [ i.info("initializing in 8 bit") for i in [qa_rag,qa_chat]]
        #info = qa_chat.info("initializing in 8bit")
        bnb_config = BitsAndBytesConfig(transformers.BitsAndBytesConfig(
        load_in_8bit=True))
        for i in infos:
            i.empty()

    elif mode == "4bit":
        st.session_state.startup = False
        st.session_state.current_qbit_mode = "4bit"

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
        start_time = time.time()
        data: ChatBotResponseModel = st.session_state.ic.ask_me_something(input_text)
        inference_time = time.time() - start_time
    model_use = data.model_use
    qbm = st.session_state.current_qbit_mode

    add_row(data.answer,"rows")
    add_row(f"`{datetime.datetime.now()}`- exec time: `{inference_time}s` - llm model: `{model_use}` - quatization bitmode: `{qbm}`", "rows")
    add_row("="*50, "rows")

    info = st.info("response generated")
    time.sleep(1)
    info.empty()

def chat_rag(input_text):
    with st.spinner("asking llm"):
        start_time = time.time()
        data: ChatBotResponseModel = st.session_state.ic.ask_with_rag(input_text)
        inference_time = time.time() - start_time
    model_use = data.model_use
    qbm = st.session_state.current_qbit_mode
    add_row(data.answer,"rows_rag")
    add_row(f"`{datetime.datetime.now()}`- exec time: `{inference_time}s` - llm model: `{model_use}` - quatization bitmode: `{qbm}`", "rows_rag")
    add_row("="*50, "rows_rag")

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

def vector_load_from_file(filename: str):
    vector_tab.info(f"populating vector db with content from {filename}")
    st.session_state.ic.load_text_from_local(filename)
    return None

def transcribe(file: Path):
    with st.spinner(f"transcribing file {file.name}"):
        start = time.time()
        data: AudioDataReadModel = st.session_state.audio_task.transcribe(file.as_posix())
        inference_time = time.time() - start
    add_row(data.text,"rows_audio")
    add_row(f"`{datetime.datetime.now()}`- exec time: `{inference_time}s` - audio model: `{data.model}` - file : `{file}`", "rows_audio")
    add_row("="*50, "rows_audio")

def log(line: str, widget):
    widget.write(line)

### MAIN section
#
#sider
init_8bit_button = st.sidebar.button('Initialized 8bit')
init_4bit_button = st.sidebar.button('Initialized 4bit')
get_history = st.sidebar.button('show memory')

if st.session_state.startup:
    llm_init = True
    audio_init = True

    if llm_init:
        ic, llm, cb_ctr = initialiaze_model("4bit")
        st.session_state.ic = ic
        st.session_state.llm = llm
        st.session_state.cb_ctr = cb_ctr

    if audio_init:
        wds, audio_ctr, audio_task = initialize_audio()

        st.session_state.audio_task = audio_task
        st.session_state.audio_ctr = audio_ctr
        st.session_state.whisper_ds = wds



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


### QA tab section
#
with col1_chat.form('chat'):
        text = st.text_area('Ask me something:', 'give me a list of animals?')
        submitted = st.form_submit_button('Submit')
        if submitted:
            chat(text)
chat_content = qa_chat.columns(1)[0]

for data in st.session_state["rows"][::-1]:
    row_data = generate_row_chat(data, chat_content)
    rows_collection_chat.append(row_data)

### QA rag tab section
#
with col2_chat_rag.form('chat_rag'):
        text = st.text_area('RAG - Ask me something:', 'What is cloudera cml')
        submitted = st.form_submit_button('Submit')
        if submitted:
            chat_rag(text)

chat_rag_content = qa_rag.columns(1)[0]

for data in st.session_state["rows_rag"][::-1]:
    row_data = generate_row_chat(data, chat_rag_content)
    rows_collection_chat_rag.append(row_data)


### VECTORS section
#
col_url, col_upload = vector_tab.columns([0.8, 0.2]) 


with col_url.form("url-form"):
    urls = "https://docs.cloudera.com/machine-learning/cloud/index.html,https://docs.cloudera.com/machine-learning/cloud/product/topics/ml-product-overview.html#cdsw_overview"
    text = st.text_area('Pass a list of comma separated urls:', urls)
    submitted = st.form_submit_button('Submit')
    if submitted:
        vector_load_from_web(text)

uploader = col_upload.file_uploader("choose text file(s) to upload", accept_multiple_files=True)

for uploaded_file in uploader:
    filename = uploaded_file.name
    trgt_path = f"tmp-data/text/{filename}"
    with open(trgt_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    info = vector_tab.info(f"file {filename} uploaded")
    vector_load_from_file(trgt_path)
    info.empty()




### AUDIO tab section
#
audio_tab_main, audio_tab_right = audio_tab.columns([0.8, 0.2])


with audio_tab_right.form("audio-uploader-form", clear_on_submit=True):
    audio_uploader = st.file_uploader("choose audio file(s) to upload (wav, flac or mp3)", accept_multiple_files=True)
    submitted = st.form_submit_button("UPLOAD")
    
    if submitted and audio_uploader is not None:
        for uploaded_file in audio_uploader:
            filename = uploaded_file.name
            trgt_path = Path(f"tmp-data/audio/{filename}")
            error_text = color_text("red", "ERROR")
            if trgt_path.suffix not in (".wav",".flac",".mp3"):
                add_row(f":red[ERROR] - :file {filename} is no a supported format","rows_audio_log")
                continue

            st.session_state.audio_files.append(trgt_path)
            with open(trgt_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            add_row(f"`INFO` - file {filename} uploaded", "rows_audio_log")


checkbox_labels = {p for p in st.session_state.audio_files}
option = audio_tab_main.selectbox(
    'Choose and audio file to be played',
    checkbox_labels)

if option != None:
    path = Path(option)
    info = audio_tab_main.info(f"selected: {path.name}")

    audio_file = open(option, 'rb')
    audio_bytes = audio_file.read()
    audio_tab_main.audio(audio_bytes, format=f"audio/{path.suffix}")

    audio_tab_main.button("Transcribe", type="primary" ,on_click=transcribe,args=(path,))
    audio_tab_main.button("Cancel", type="secondary")

    info.empty()

for data in st.session_state["rows_audio"][::-1]:
    row_data = generate_row_chat(data, audio_tab_main)
    rows_collection_audio.append(row_data)

for data in st.session_state["rows_audio_log"][::-1]:
    row_data = generate_row_chat(data, audio_tab_right)
    rows_collection_audio.append(row_data)


### CSS section
#
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)