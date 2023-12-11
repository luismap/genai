import asyncio
from itertools import groupby
from typing import Awaitable, List, Tuple
import logging
import yaml
from core.utils.MyUtils import MyUtils
import concurrent

from features.chatbot.data.datasource.WhisperDataSource import WhisperDataSource
from features.chatbot.data.controller.AudioController import AudioController
from features.chatbot.domain.usecase.AudioTask import AudioTask
from features.chatbot.data.models.AudioDataModel import AudioDataPayloadModel, AudioDataReadModel

wds = WhisperDataSource()
audio_ctr = AudioController([wds])
audio_task = AudioTask(audio_ctr)


queue: asyncio.Queue = None
global_loop: asyncio.AbstractEventLoop = None
model = ""
max_batch_size = 9
wait_time = 2.0

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

def group_by_task(audio_payloads:List[Tuple[Awaitable[AudioDataReadModel], AudioDataPayloadModel]]
                           ) -> List[List[Tuple[Awaitable[AudioDataReadModel], AudioDataPayloadModel]]]:
        keyfunc = lambda audio_payload_model: audio_payload_model[1].task
        grouped = []
        for _, groups in groupby(audio_payloads, keyfunc):
            grouped.append(list(groups))

        return grouped

async def batch_ask(adpm: AudioDataPayloadModel) -> Awaitable[AudioDataReadModel]:
	if not global_loop:
		raise ValueError("The global loop has not initialized for translation.")

	job_future: concurrent.Future[AudioDataReadModel] = global_loop.create_future()

	logger.info(f"my future type: {type(job_future)}")	
	await queue.put((job_future, adpm))
	
	return await job_future

#pool_model = concurrent.futures.ProcessPoolExecutor(max_workers=1, initializer=initialize_model)

def process_batch(batch: List[Tuple[Awaitable[AudioDataReadModel], AudioDataPayloadModel]]) -> None:
	
	task_batch = group_by_task(batch)

	for task_group in task_batch:
		print(task_group)
		task_type = task_group[0][1].task
		jobs_future, adpms = zip(*task_group)

		logger.info(f"object before asking llm: {type(adpms)}")
		if task_type == 'transcribe':
			response = audio_task.transcribe(adpms)
		elif task_type == 'translate':
			response = audio_task.transcribe(adpms)

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
			current_batch: List[Tuple[Awaitable[AudioDataReadModel],AudioDataPayloadModel]] = [await queue.get()]

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