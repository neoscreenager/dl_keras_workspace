# import gradio as gr
# import torch
# #from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import pipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# def ask(text):
#
#   #model = AutoModelForCausalLM.from_pretrained("/home/neo/local_llm_models/Publisher/Repository/llama-2-7b-chat.Q4_K_M.gguf")
#   model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="/home/neo/local_llm_models/Publisher/Repository/llama-2-7b-chat.Q4_K_M.gguf", model_type="llama", gpu_layers=1)
#   #tokenizer = AutoTokenizer.from_pretrained(model)
#   pipe = pipeline("conversational", model=model)    
#
#   #inputs = tokenizer(text, return_tensors='pt', return_token_type_ids=False).to(model.device)
#
#   #input_length = inputs.input_ids.shape[1]
#   #outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.9)
#   #outputs = model(**inputs)  
#
#   #tokens = outputs.sequences[0, input_length:]
#   return pipe(text, max_new_tokens=256)
#   #return tokenizer.decode(tokens)
#
# with gr.Blocks() as server:
#   with gr.Tab("LLM Inferencing"):
#     model_input = gr.Textbox(label="Your Question:", 
#                              value="What’s your question?", interactive=True)
#     ask_button = gr.Button("Ask")
#     model_output = gr.Textbox(label="The Answer:", 
#                               interactive=False, value="Answer goes here...")
#
#   ask_button.click(ask, inputs=[model_input], outputs=[model_output])
#
# server.launch()

# from ctransformers import AutoModelForCausalLM, AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained(
# "TheBloke/Llama-2-7b-Chat-GGUF",
# #model_file="llama-2-7b-chat.Q4_K_M.gguf",
# #model_type="llama",
# gpu_layers=1
# )
# print(model("What is AI"))


# import gradio as gr
# from transformers import pipeline
# from ctransformers import AutoModelForCausalLM, AutoTokenizer
#
# def ask(text):
#     # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
#     #llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.q4_K_M.gguf", model_type="llama")
#     llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF")
#     return llm(text)
#
# with gr.Blocks() as server:
#   with gr.Tab("LLM Inferencing"):
#     model_input = gr.Textbox(label="Your Question:", 
#                              value="What’s your question?", interactive=True)
#     ask_button = gr.Button("Ask")
#     model_output = gr.Textbox(label="The Answer:", 
#                               interactive=False, value="Answer goes here...")
#
#   ask_button.click(ask, inputs=[model_input], outputs=[model_output])
#
# server.launch()

from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf" # meta-llama/Llama-2-7b-chat-hf

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
from transformers import pipeline

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

SYSTEM_PROMPT = """<s>[INST] <<SYS>>
Limit your responses to hospital's operations and health services provided by Apollo Hospitals.
<</SYS>>

"""

def get_response(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256,
    )
    print("Chatbot:", sequences[0]['generated_text'])



# Formatting function for message and history
def format_message(message: str, history: list, memory_limit: int = 3) -> str:
    """
    Formats the message and history for the Llama model.

    Parameters:
        message (str): Current message to send.
        history (list): Past conversation history.
        memory_limit (int): Limit on how many past interactions to consider.

    Returns:
        str: Formatted message string
    """
    # always keep len(history) <= memory_limit
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        return SYSTEM_PROMPT + f"{message} [/INST]"

    formatted_message = SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"

    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"

    # Handle the current message
    formatted_message += f"<s>[INST] {message} [/INST]"

    return formatted_message

# Generate a response from the Llama model
def get_llama_response(message: str, history: list) -> str:
    """
    Generates a conversational response from the Llama model.

    Parameters:
        message (str): User's input message.
        history (list): Past conversation history.

    Returns:
        str: Generated response from the Llama model.
    """
    query = format_message(message, history)
    response = ""

    sequences = llama_pipeline(
        query,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,
    )

    generated_text = sequences[0]['generated_text']
    response = generated_text[len(query):]  # Remove the prompt from the output

    print("Chatbot:", response.strip())
    return response.strip()

import gradio as gr

gr.ChatInterface(get_llama_response).launch()

