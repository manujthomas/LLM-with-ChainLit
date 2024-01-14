import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from dotenv import load_dotenv
import os

model_id = "tiiuae/falcon-7b-instruct"
#model_id = "tiiuae/falcon-40b-instruct"
#model_id = "WizardLM/WizardLM-13B-V1.2"

llm = HuggingFaceHub(
	repo_id = model_id,
	model_kwargs = {"temperature":0.3, "max_new_tokens":512},
	#huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
	huggingfacehub_api_token = "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"
)

template = """
	You are an AI assistant that provides helpful answers to user queries.

	{question}

"""

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # Send the response
    await cl.Message(content=res["text"]).send()