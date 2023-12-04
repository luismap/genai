import asyncio
from typing import Awaitable, List, Tuple
import logging
import yaml
from core.utils.MyUtils import MyUtils
import concurrent

from core.llm.models.configs.BitsAndBytes import BitsAndBytesConfig
from features.chatbot.data.datasource.RagLlama2DataSource import RagLlama2DataSource
from features.chatbot.data.datasource.FaissLangchainVectorDbSource import FaissLangchainVectorDbSource
from features.chatbot.data.models.ChatRagModel import ChatRagPayloadModel, ChatRagReadModel
from features.chatbot.data.controller.ChatRagController import ChatRagController
from features.chatbot.domain.usecase.RagInteractiveChat import RagInteractiveChat


bnb_config = BitsAndBytesConfig()
vector_db = FaissLangchainVectorDbSource()
llm_source = RagLlama2DataSource(vector_db, bnb_config)
cb_ragctr = ChatRagController([llm_source])
rag_uc = RagInteractiveChat(cb_ragctr)

queue: asyncio.Queue = None
global_loop: asyncio.AbstractEventLoop = None
model = ""
max_batch_size = 9
wait_time = 3.0

canlog = True
appProps = MyUtils.load_properties("general")["app"]

with open("logging.yaml", 'rt') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

# Get an instance of the logger and use it to write a log!
# Note: Do this AFTER the config is loaded above or it won't use the config.
logger = logging.getLogger(appProps["logger"])
logger.info("Initial log config in route post!")

#core setup for async
language_code_map = ["english", "spanish"]

async def batch_ask(crpm: ChatRagPayloadModel) -> Awaitable[ChatRagReadModel]:
	if not global_loop:
		raise ValueError("The global loop has not initialized for translation.")

	job_future: concurrent.Future[ChatRagReadModel] = global_loop.create_future()

	logger.info(f"my future type: {type(job_future)}")	
	await queue.put((job_future, crpm))
	
	return await job_future


async def vector_search(vector):
    # simulate I/O call (e.g. Vector Similarity Search using a VectorDB)
    await asyncio.sleep(1.005)

def initialize_model():
	global model
	model = "33"

def clean_user_context(user_id: str) -> bool:
	return rag_uc.clean_context(user_id=user_id)

#pool_model = concurrent.futures.ProcessPoolExecutor(max_workers=1, initializer=initialize_model)

def process_batch(batch: List[Tuple[Awaitable[ChatRagReadModel], ChatRagPayloadModel]]) -> None:
	jobs_future, crpms = zip(*batch)

	logger.info(f"object before asking llm: {type(crpms)}")
	response = rag_uc.ask(crpms)
	
	for idx in range(len(response)):
		logger.info(f"object before sending future: {type(response[idx])}")
		jobs_future[idx].set_result(response[idx])
		queue.task_done()

async def batch_processing_loop(loop: asyncio.AbstractEventLoop):
	global queue
	global global_loop
	
	if not queue:
		logger.debug("Setting the global queue and async loop.")
		queue = asyncio.Queue(loop=loop, maxsize=max_batch_size)
		global_loop = loop

	#with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
	with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
		while True:
			logger.info("Starting a batch processing")
			current_batch: List[Tuple[Awaitable[ChatRagReadModel],ChatRagPayloadModel]] = [await queue.get()]

			while (
				len(current_batch) < max_batch_size 
				and not queue.empty()
			):
				try:
					current_batch.append(await asyncio.wait_for(
						queue.get(), wait_time
					))
				except TimeoutError:
					logger.info("batch timeout")
					break
				
			logger.info(f"Processing batch of size {len(current_batch)}")
			await loop.run_in_executor(pool, process_batch, current_batch)