
#pip install -q transformers einops accelerate langchain bitsandbytes

from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch

model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=400,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0.2})

#!pip install chainlit

from langchain import PromptTemplate,  LLMChain
import chainlit as cl

template = """
You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Question: {question}
Answer:"""

#prompt = PromptTemplate(template=template, input_variables=["question"])
#llm_chain = LLMChain(prompt=prompt, llm=llm)
#question = "Which are neighbor countries of India"
#print(llm_chain.run(question))

@cl.langchain_factory(use_async=False)
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain
