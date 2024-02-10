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

llm = LlamaCPP(
    model_path="/home/neo/local_llm_models/Publisher/Repository/llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.8, #increased the temprature so that LLM will give a creative response
    max_new_tokens=500,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers":1},
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
# load documents
documents = SimpleDirectoryReader(
    "./data"
).load_data()
# create vector store index
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
# set up query engine
query_engine = index.as_query_engine()
response = query_engine.query("How to replace hard disk on my Lenovo laptop?")
print(response)
#response = llm.complete("Hi, can you generate a python code to upload files?")
#print(response.text)
# response_iter = llm.stream_complete("Who is the founder of Apollo Hospital?")
# #stream response
# for response in response_iter:
#     print(response.delta,end="",flush=True)
    