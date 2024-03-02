'''
    Running locally downloaded LLAMA-2-7B quantized LLM model (in gguf format),
    using LLAMA INDEX and LLAMA CPP.
    Program uses a chat interface using Streamlit
    
    To install before running this program:
    pip install llama-index-embeddings-huggingface
    pip install llama-index-llms-llama-cpp
    pip install llama-index
    pip install transformers
    pip install streamlit
    
'''
import streamlit as st
from git import index
# moved the set_page_config to the top so that it will be the first Stremalit line to execute
# otherwise it was throwing a StreamlitAPIException
st.set_page_config(
        page_title="Personal Chatbot"
    )
st.header("Persoanl Chatbot")
st.sidebar.title("Options")
    
from langchain.schema import(SystemMessage, HumanMessage, AIMessage)

from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
     )
from llama_index.llms.llama_cpp import LlamaCPP

from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)


def select_llm() -> LlamaCPP:
    return LlamaCPP(
    model_path="/home/neo/local_llm_models/Publisher/Repository/llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.3, #increased the temperature so that LLM will give a creative response
    max_new_tokens=500,
    context_window=3900,
    generate_kwargs={},
    #model_kwargs={"n_gpu_layers":1}, #comment this line if running only on CPU
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
    )

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation",key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant created by Apollo Hospitals. Reply your answers in markdown format"
                          )
            ]

def get_answers(llm,messages) -> str:
    response = llm.complete(messages)
    return response.text

def init_vector_index() -> VectorStoreIndex:
    # change the global tokenizer to match our LLM.
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
        )
    # use Huggingface embeddings
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # load documents from "data" folder
    documents = SimpleDirectoryReader("./data").load_data()
    # create vector store index
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)

def get_answers_from_vector_index(index,llm,user_input) -> str:
    query_engine = index.as_query_engine(llm=llm,streaming=False,similarity_top_k=1)
    streaming_response = query_engine.query(user_input)
    return str(streaming_response)
        
def main() -> None:
    #init_page()
    llm = select_llm()
    index = init_vector_index()
    init_messages()
    
    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Bot is typing..."):
            answer = get_answers_from_vector_index(index,llm,user_input) #get_answers(llm, user_input)
            print(answer)
        st.session_state.messages.append(AIMessage(content=answer))
        #st.session_state.messages.append(AIMessage(content=get_answers_from_vector_index(index, llm,user_input) )) 
        
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)



if __name__ == "__main__":
    main()
    
    

