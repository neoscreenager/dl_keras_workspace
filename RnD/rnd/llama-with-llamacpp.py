'''
    Running locally downloaded LLAMA-2-7B quantized LLM model (in gguf format),
    using LLAMA INDEX and LLAMA CPP.
    
    To install before running this program:
    pip install llama-index-embeddings-huggingface
    pip install llama-index-llms-llama-cpp
    pip install llama-index
    pip install transformers
    
'''

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

llm = LlamaCPP(
    model_path="/home/neo/local_llm_models/Publisher/Repository/llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.3, #increased the temperature so that LLM will give a creative response
    max_new_tokens=500,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers":1}, #comment this line if running only on CPU
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
    )

# response_iter = llm.stream_complete("Who is the founder of Apollo Hospital?")
# #stream response
# for response in response_iter:
#     print(response.delta,end="",flush=True)
    
'''
Query engine setup with LlamaCPP

'''

from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer
# change the global tokenizer to match our LLM.

set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# use Huggingface embeddings

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# load documents from "data" folder
documents = SimpleDirectoryReader(
    "./data"
).load_data()

# create vector store index
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# set up query engine
query_engine = index.as_query_engine(llm=llm,streaming=True, similarity_top_k=1)

streaming_response = query_engine.query("What is the process of changing hard disk of lenovo laptop?")
streaming_response.print_response_stream()

# # create vector store index    
# index = VectorStoreIndex.from_documents(
#     documents, 
# )
# index.storage_context.persist() # persisting the vector indexes in json format in "storage" folder in current directory
# # set up query engine
# query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
# # To use the already index vector database:
# from llama_index.core import StorageContext,load_index_from_storage
# storage_context = StorageContext.from_defaults(persist_dir="./storage")
# index = load_index_from_storage(storage_context=storage_context)
#
# response = query_engine.query("how to change the battery of lenovo laptop?")
# response.print_response_stream()
#print(response)
#response = llm.complete("Hi, can you generate a python code to upload files?")
#print(response.text)
# response_iter = llm.stream_complete("Who is the founder of Apollo Hospital?")
# #stream response
# for response in response_iter:
#     print(response.delta,end="",flush=True)
    