from vllm import LLM, SamplingParams


prompts_long = [
    "Hello, my name is",
    "what is cloudera cml",
    "what do you know about cloudera?",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "What country has the highest life expectancy?",
    "Where would you be if you were standing on the Spanish Steps?",
    "Which language has the more native speakers: English or Spanish?",
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "What country has the highest life expectancy?",
    "Where would you be if you were standing on the Spanish Steps?",
    "Which language has the more native speakers: English or Spanish?",
    "What is the 4th letter of the Greek alphabet?",
    "What sports car company manufactures the 911?",
    "What city is known as The Eternal City?",
    "Roald Amundsen was the first man to reach the South Pole, but where was he from?",
    "What is the highest-rated film on IMDb as of January 1st, 2022?",
    "Who discovered that the earth revolves around the sun?",
]

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.2, top_p=0.95)


from huggingface_hub import login
import os


token = os.getenv("hf_token")

login(token)


model_id = "mistralai/Mistral-7B-Instruct-v0.1"
# model_id = "meta-llama/Llama-2-7b-chat-hf"

llm = LLM(model_id, tensor_parallel_size=2, dtype="half")

# llm = LLM(model_id, tensor_parallel_size=4, dtype="half")


ans = llm.generate(prompts, sampling_params)


def get_questions():
    with open("random_questions.txt", "r") as file:
        questions = []
        for q in file.readlines():
            questions.append(q)

        return questions


def pp(ans):
    for r in ans:
        output = r.outputs[0]
        print("=" * 100)
        print(f"{r.prompt}: \ {output.text}")


pp(ans)
