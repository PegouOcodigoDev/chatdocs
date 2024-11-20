import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import time
import os
from retriever import config_retriever
from chain import configure_rag_chain

st.set_page_config(page_title="Carregue seus documentos")
st.title("Carregue seus documentos")

uploads = st.sidebar.file_uploader(
    label="Enviar arquivos", type=["pdf"],
    accept_multiple_files=True
)

if not uploads:
    st.info("por favor envie algum arquivo para continuar")
    st.stop()
    
if "history" not in st.session_state:
    st.session_state.history = [AIMessage(content="Olá, sou seu assistente virtual!")]
    
if "docs_list" not in st.session_state:
    st.session_state.docs_list = None
    
if "retriever" not in st.session_state:
    st.session_state.retriever = None

for message in st.session_state.history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

start = time.time()
user_query = st.chat_input("Digite sua mensagen aqui...")

if user_query is not None and user_query != "" and uploads is not None:
    st.session_state.history.append(HumanMessage(content=user_query))
    
    with st.chat_message("human"):
        st.markdown(user_query)
    
    with st.chat_message("ai"):
        if st.session_state.docs_list != uploads:
            st.session_state.docs_list = uploads
            st.session_state.retriever = config_retriever(uploads)
            
        rag_chain = configure_rag_chain(st.session_state.retriever)
        
        result = rag_chain.invoke({"input": user_query, "history": st.session_state.history})
        
        answer = result['answer']
        st.write(answer)
        
        sources = result['context']
        for idx, doc in enumerate(sources):
            source = doc.metadata['source']
            file = os.path.basename(source)
            page = doc.metadata.get('page', 'pagina não especificada')
            
            ref = f":link: Fonte {idx + 1}: *{file} -p. {page}*"
            with st.popover(ref):
                st.caption(doc.page_content)
    st.session_state.history.append(AIMessage(content=answer))
    
end = time.time()
print("tempo de resposta:", end - start)