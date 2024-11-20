from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from model import FactoryModel


def configure_tokens(model_type):
    if model_type.startswith("hf"):
        token_s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        token_e = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        token_s, token_e = "", ""
    return token_s, token_e


def create_context_question_prompt(system_prompt , token_s, token_e):
    system_prompt = token_s + system_prompt
    user_prompt = "Question: {input}" + token_e

    return ChatPromptTemplate.from_messages(
        [("system", system_prompt),
         MessagesPlaceholder("history"),
         ("human", user_prompt)]
    )

def configure_rag_chain(retriever, model_type="hf_hub"):
    token_s, token_e = configure_tokens(model_type)
    
    system_prompt = (
        "Given the following chat history and the follow-up question which might reference "
        "the chat history, formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as it."
    )
    context_prompt = create_context_question_prompt(system_prompt, token_s, token_e)
    
    llm = FactoryModel(model_type=model_type)
    llm = llm.get_model()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    template = """
    Você é um assistente prestativo e está respondendo perguntas gerais. Responda as perguntas em português.
    Use os seguintes pedaços de texto para responder as perguntas.
    Se você não sabe a resposta, apenas diga que não saiba. Mantenha a resposta concisa.
    Responda em português.
    \n\n
    Pergunta:{input}\n
    Contexto:{context}
    """
    qa_prompt = PromptTemplate.from_template(token_s + template + token_e)

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_cahin = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_cahin
