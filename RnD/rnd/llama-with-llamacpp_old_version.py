from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
     )
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
    )
from llama_index.storage import storage_context

llm = LlamaCPP(
    model_path="/home/neo/local_llm_models/Publisher/Repository/llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.3, #increased the temprature so that LLM will give a creative response
    max_new_tokens=500,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers":1}, #comment this line if running on cpu only
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
    )

from llama_index import set_global_tokenizer
from transformers import AutoTokenizer

set_global_tokenizer(
    AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
)

# use Huggingface embeddings
from llama_index.embeddings import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# create a service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)   
# load documents from "data" folder
documents = SimpleDirectoryReader(
    "./data"
).load_data()
# create vector store index    
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
index.storage_context.persist() # persisting the vector indexes in json format in "storage" folder in current directory
# set up query engine
query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
# To use the already index vector database:
# from llama_index import StorageContext,load_index_from_storage
# storage_context = StorageContext.from_defaults(persist_dir="./storage")
# index = load_index_from_storage(storage_context=storage_context)

response = query_engine.query("how to change the battery of lenovo laptop?")
response.print_response_stream()
#print(response)
#response = llm.complete("Hi, can you generate a python code to upload files?")
#print(response.text)
# response_iter = llm.stream_complete("Who is the founder of Apollo Hospital?")
# #stream response
# for response in response_iter:
#     print(response.delta,end="",flush=True)
    