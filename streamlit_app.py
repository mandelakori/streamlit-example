import streamlit as st
import torch
from transformers import pipeline, set_seed

# Set up the text generation pipeline
pipe = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-alpha",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Function to generate AI response
def generate_response(user_input):
    instruction = (
        "Your name is AISAK, which stands for 'Artificially Intelligent Swiss Army Knife'. "
        "You are built by Mandela Logan. You are Mandela Logan's first implementation of a multi-purpose AI clerk. "
        "You are an assistant, and your task is to assist the user in every query."
    )

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ]

    # Exclude the system instruction from the output
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )[1:]
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    # Extract and return the generated text
    return outputs[0]["generated_text"]

# Streamlit UI
st.title("AISAK - AI Chatbot")

user_input = st.text_input("You:")
if st.button("Send"):
    if user_input.lower() in ["exit", "quit", "bye"]:
        st.write("AISAK: Goodbye!")
    else:
        response = generate_response(user_input)
        st.write(f"AISAK: {response}")
