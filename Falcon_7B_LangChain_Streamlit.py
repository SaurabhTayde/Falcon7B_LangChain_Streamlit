import streamlit as st
st.title("Falcon 7B Instruct Bot")

#pip install -q transformers einops accelerate langchain bitsandbytes

from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
from langchain import PromptTemplate,  LLMChain

model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=600,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0.2})

template = """
You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, verbose=True, llm=llm)

def chat(query):
    result = llm_chain.predict(question=query)
    return result

def main():
    input = st.text_input("How can I Help You?", placeholder="Please ask your question here")
    if input:
        output = chat(input)
        st.write(output,unsafe_allow_html=True)

if __name__ == '__main__':
    main()




#--------------------------------


#!pip install chainlit
#import chainlit as cl

# prompt_template = "{input}?"
#
# @cl.langchain_factory(use_async=False)
# def main():
#     chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
#     return chain
#
# @cl.langchain_run
# async def run(agent, input_str):
#     res = await cl.make_async(agent)(input_str, callbacks=[cl.ChainlitCallbackHandler()])
#     await cl.Message(content=res["text"]).send()


#
# @cl.langchain_factory(use_async=False)
# def factory():

#
#     return llm_chain
