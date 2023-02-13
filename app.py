import torch

import gradio as gr
from transformers import pipeline

MODEL_NAME = "openai/whisper-large-v2"
#MODEL_NAME = "openai/whisper-medium"
#MODEL_NAME = "mitchelldehaven/whisper-large-v2-ru"

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=29.9,
    device="cpu",
)

all_special_ids = pipe.tokenizer.all_special_ids
transcribe_token_id = all_special_ids[-5]
translate_token_id = all_special_ids[-6]

def transcribe(file_upload, task):
    warn_output = ""
    if file_upload is None:
        return "ERROR: You have to upload an audio file"

    file = file_upload

    pipe.model.config.forced_decoder_ids = [[2, transcribe_token_id if task=="transcribe" else translate_token_id]]

    text = pipe(file)["text"]

    return warn_output + text

demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="upload", type="filepath", optional=True),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
    ],
    outputs="text",
    layout="horizontal",
    theme="huggingface",
    #title="Whisper Large V2: Transcribe Audio",
    description=(
        "Server uses"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe], ["Transcribe Audio"])

demo.launch(enable_queue=True, share=True)

