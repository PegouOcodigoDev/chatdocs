from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

class Model:
    load_dotenv()
    
    def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
        llm = HuggingFaceHub(
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        repo_id=model,
        model_kwargs={
            "temperature": temperature,
            "return_full_text":False,
            "max_new_tokens": 512,
        })
        return llm

    def model_ollama(model="phi3",temperature=0.1):
        llm = ChatOllama(model=model, temperature=temperature)
        return llm

    def model_openai(model="gpt-4o-mini", temperature=0.1):
        llm = ChatOpenAI(model=model, temperature=temperature)
        return llm
    
class FactoryModel:
    def __init__(self, model_type="hf_hub", **kwargs):
        self.model_type = model_type

    def get_model(self,**kwargs):
        match self.model_type:
            case "hf_hub":
                return Model.model_hf_hub(**kwargs)
            case "ollama":
                return Model.model_ollama(**kwargs)
            case "openai":
                return Model.model_openai(**kwargs)
            case _:
                raise ValueError(f"Tipo de modelo desconhecido: {self.model_type}")
