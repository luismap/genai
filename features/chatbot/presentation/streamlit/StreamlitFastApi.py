import json
import streamlit as st
from features.chatbot.data.controller.AudioController import AudioController
from features.chatbot.data.datasource.WhisperDataSource import WhisperDataSource
from features.chatbot.data.models.ChatBotModel import ChatBotPayloadModel, ChatBotReadModel, ChatBotResponseModel
from features.chatbot.data.models.ChatRagModel import ChatRagPayloadModel, ChatRagReadModel, ChatRagResponseModel
from features.chatbot.domain.usecase.InteractiveChat import InteractiveChat
from features.chatbot.domain.usecase.AudioTask import AudioTask
from features.chatbot.data.models.AudioDataModel import AudioDataPayloadModel, AudioDataReadModel, AudioDataResponseModel
import requests
import datetime
import time
import pandas as pd
from pathlib import Path
import uuid

#create session state for rows
if "rows" not in st.session_state:
    st.session_state["rows"] = []
if "rowsqa_logs" not in st.session_state:
    st.session_state["rowsqa_logs"] = []
if "rows_rag" not in st.session_state:
    st.session_state["rows_rag"] = []
if "rows_rag_log" not in st.session_state:
    st.session_state["rows_rag_log"] = []
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
if "user_name" not in st.session_state:
    st.session_state.user_name = str(uuid.uuid1())
if "qa_history" not in st.session_state:
    st.session_state.qa_history = False
if "rag_history" not in st.session_state:
    st.session_state.rag_history = False
if "urls_list" not in st.session_state:
    st.session_state.urls_list = False
if "is_web_upload" not in st.session_state:
    st.session_state.is_web_upload = False

rows_collection_chat = []
rows_collection_chat_logs = []
rows_collection_chat_rag = []
rows_collection_audio = []
rows_collection_audio_log = []

#vars
qa_url = 'https://public-1bf4beb4a6taubdd.ml-d546e9a6-a5b.se-sandb.a465-9q4k.cloudera.site/'
rag_url = 'https://public-67agq5by0c0i8fua.ml-d546e9a6-a5b.se-sandb.a465-9q4k.cloudera.site/'
audio_url = 'https://public-6t327a3iudrcisqn.ml-d546e9a6-a5b.se-sandb.a465-9q4k.cloudera.site/'

#page configs
st.set_page_config(layout="wide")
st.title('📱Gen - AI © PS labs')

#initialized widgets
qa_chat, qa_history, qa_rag, rag_history_tab, vector_tab, audio_tab = st.tabs(["📝 QA", "📚 QA History", "🎓 QA rag", "📚 Rag History" , "🧮 vectors", "🔈 audio"])
username_chat, history_chat = qa_chat.columns(2)
col1_chat = qa_chat.columns(1)[0]
chat_content = qa_chat.columns(1)[0]

col2_chat_rag = qa_rag.columns(1)[0]
chat_rag_content = qa_rag.columns(1)[0]

#functional scripts

def color_text(color: str, text: str):
    text = f"<span style=\"color:{color}\">{text}</span>"
    return text

def add_row(content,key):
    st.session_state[key].append(content)

def add_row_qa_log(key,content):
    st.session_state[key]= [content]

def remove_row(content,key):
    st.session_state[key].remove(content)

def generate_row_chat(content, widget):
    with widget:
        widget.write(content)
    return content


def chat(input_text):
    route = 'qabot/ask'
    question = input_text
    user_id = st.session_state.user_name
    history = st.session_state.qa_history
    payload = ChatBotPayloadModel(user_id=user_id,question=question, history=history)
    
    with st.spinner("asking llm"):
        start_time = time.time()
        r = requests.post(qa_url+route, data=payload.json(), headers={'Content-Type': 'application/json'})
        inference_time = time.time() - start_time
        data = ChatBotReadModel.parse_raw(json.dumps(r.json()))
        model_use = data.model_use
        #print(data)
        qbm = st.session_state.current_qbit_mode

    if history:
        add_row_qa_log("rowsqa_logs",data.chat_history)

    add_row(data.answer,"rows")
    add_row(f"`{datetime.datetime.now()}`- exec time: `{inference_time}s` - llm model: `{model_use}` - quatization bitmode: `{qbm}`", "rows")
    add_row("="*50, "rows")

    info = st.info("response generated")
    time.sleep(1)
    info.empty()

def chat_rag(input_text):
    route = 'rag/ask-rag'
    question = input_text
    user_id = st.session_state.user_name
    history = st.session_state.rag_history
    #print(f"rag history {history}")
    payload = ChatRagPayloadModel(user_id=user_id,question=question, history=history)
    #print(payload.json())
    with st.spinner("asking llm"):
        start_time = time.time()
        r = requests.post(rag_url+route, data=payload.json(), headers={'Content-Type': 'application/json'})
        inference_time = time.time() - start_time
        data =  ChatRagResponseModel.parse_raw(json.dumps(r.json()))
        model_use = data.model_use
        qbm = st.session_state.current_qbit_mode

    #print(data)
    if history:
        add_row_qa_log("rows_rag_log",data.chat_history)

    add_row(data.answer,"rows_rag")
    add_row(f"`{datetime.datetime.now()}`- exec time: `{inference_time}s` - llm model: `{model_use}` - quatization bitmode: `{qbm}`", "rows_rag")
    add_row("="*50, "rows_rag")

def generate_web_button(urls):
    route = 'rag/web-url-src-upload'    
    
    r = requests.post(rag_url+route, data=json.dumps(urls), headers={'Content-Type': 'application/json'})
    

def vector_load_from_web(content: str):
    urls = content.strip()
    urls_parsed = urls.strip(",")
    urls_list = urls_parsed.split(",")
    vector_tab.warning(f"Following links parsed",icon="ℹ️")
    data = pd.DataFrame(urls_list,columns=["url"])
    vector_tab.dataframe(data, 
                          column_config= {"url": st.column_config.LinkColumn(" 🔗 Links Parsed")}
                          ,hide_index=True)
    vector_tab.button("Send", type="primary" ,on_click=generate_web_button,args=(urls_list,), key="url_send")
    vector_tab.button("Cancel", type="secondary")

def vector_load_from_file(filename: str):
    info = vector_tab.info(f"populating vector db with content from {filename}")
    route = f"rag/document-upload?path={filename}"
    r = requests.post(rag_url+route)
    info.empty() 
    return r.json()

def transcribe(file: Path, language: str):
    full_name = str(file)
    payload = AudioDataPayloadModel(source_audio=full_name, 
                                    language=language,
                                    task="transcribe")
    route = "audio/transcribe"
    start = time.time()
    response  = requests.post(audio_url+route,
                              data=payload.json(),
                              headers={'Content-Type': 'application/json'})
        #print(response.json())
    data =  AudioDataResponseModel.parse_raw(json.dumps(response.json())) 
    inference_time = time.time() - start

    add_row(data.text,"rows_audio")
    add_row(f"`{datetime.datetime.now()}`- exec time: `{inference_time}s` - audio model: `{data.model}` - file : `{file}`", "rows_audio")
    add_row("="*50, "rows_audio")

def translate(file: Path, language: str):
    full_name = str(file)
    payload = AudioDataPayloadModel(source_audio=full_name,
                                    language=language,
                                    task="translate")
    route = "audio/translate"
    start = time.time()
    response  = requests.post(audio_url+route,
                              data=payload.json(),
                              headers={'Content-Type': 'application/json'})
        #print(response.json())
    data =  AudioDataResponseModel.parse_raw(json.dumps(response.json()))
    inference_time = time.time() - start

    add_row(data.text,"rows_audio")
    add_row(f"`{datetime.datetime.now()}`- exec time: `{inference_time}s` - audio model: `{data.model}` - file : `{file}`", "rows_audio")
    add_row("="*50, "rows_audio")

def log(line: str, widget):
    widget.write(line)

### MAIN section
#
#sider

#init_8bit_button = st.sidebar.button('Initialized 8bit')
#init_4bit_button = st.sidebar.button('Initialized 4bit')
#get_history = st.sidebar.button('show memory')

#password = text_field("Password", type="password")  # Notice that you can forward text_input parameters naturally


user_id = st.sidebar.text_input("👤 username", value=st.session_state.user_name )
st.session_state.user_name = user_id
st.sidebar.divider()

history = st.sidebar.checkbox('📚 Show QA History',st.session_state.qa_history)
st.session_state.qa_history = history

clear_qa_history = st.sidebar.button('📝 Clear QA History')

st.sidebar.divider()

rag_history = st.sidebar.checkbox('📚 Show RAG History',st.session_state.rag_history)
st.session_state.rag_history = rag_history

clear_rag_history = st.sidebar.button('📝 Clear RAG History')


if clear_qa_history:
    user_id = st.session_state.user_name
    route = f"qabot/clean-user-context?user_id={user_id}"
    r = requests.post(qa_url+route)
    st.warning(f"Cleaned context for {user_id} = {r.json()}")

if clear_rag_history:
    user_id = st.session_state.user_name
    route = f"rag/clean-user-context?user_id={user_id}"
    r = requests.post(rag_url+route)
    st.warning(f"Cleaned context for {user_id} = {r.json()}")

def get_memory():
    raise Exception("to be implemented")



### QA tab section
#
with col1_chat.form('chat'):
    text = st.text_area('Ask me something:', 'give me a list of animals? be short')
    submitted = st.form_submit_button('Submit')
    if submitted:
        chat(text)

    for data in st.session_state["rows"][::-1]:
        row_data = generate_row_chat(data, chat_content)
        rows_collection_chat.append(row_data)
    
    if st.session_state.qa_history:
        data = st.session_state.rowsqa_logs
        row_data = generate_row_chat(data, qa_history)

### QA rag tab section
#

with col2_chat_rag.form('chat_rag'):
    text = st.text_area('RAG - Ask me something:', 'What is cloudera cml?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        chat_rag(text)

    for data in st.session_state["rows_rag"][::-1]:
        row_data = generate_row_chat(data, chat_rag_content)
        rows_collection_chat_rag.append(row_data)

    if st.session_state.rag_history:
        data = st.session_state.rows_rag_log
        row_data = generate_row_chat(data, rag_history_tab)


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
    trgt_path = f"uploaded-data/text/{filename}"
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
            trgt_path = Path(f"uploaded-data/audio/{filename}")
            error_text = color_text("red", "ERROR")
            if trgt_path.suffix not in (".wav",".flac",".mp3"):
                add_row(f":red[ERROR] - :file {filename} is no a supported format","rows_audio_log")
                continue

            st.session_state.audio_files.append(trgt_path)
            with open(trgt_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            add_row(f"`INFO` - file {filename} uploaded", "rows_audio_log")

src_lang_col,translate_lang_col,_ = audio_tab_main.columns([0.2,0.2,0.6])
language = src_lang_col.text_input('🗣️ choose audio language', value="english")
translate_language = translate_lang_col.text_input('🗣️ choose translate language', value="english")

checkbox_labels = {p for p in st.session_state.audio_files}
task_labels = {'transcribe',''}
option = audio_tab_main.selectbox(
    'Choose and audio file to be played',
    checkbox_labels)

if option != None:
    path = Path(option)
    info = audio_tab_main.info(f"selected: {path.name}")

    audio_file = open(option, 'rb')
    audio_bytes = audio_file.read()
    audio_tab_main.audio(audio_bytes, format=f"audio/{path.suffix}")
    #with st.spinner(f"transcribing file {file.name}"):
    audio_tab_main.button("Transcribe", type="primary" ,on_click=transcribe,args=(path, language))
    #audio_tab_main.button("Cancel", type="secondary")
    info.empty()

if option != None:
    path = Path(option)
    info = audio_tab_main.info(f"selected: {path.name}")

    audio_file = open(option, 'rb')
    audio_bytes = audio_file.read()
    audio_tab_main.audio(audio_bytes, format=f"audio/{path.suffix}")
    #with st.spinner(f"transcribing file {file.name}"):
    audio_tab_main.button("Translate", type="primary" ,on_click=translate,args=(path, translate_language))
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