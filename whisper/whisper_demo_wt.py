#!/usr/bin/env python3
import os
import gradio as gr
import whisper
from datetime import datetime
import json

MODEL_NAME = "large-v2"
#MODEL_NAME = "large-v1"
LANG = "ru"
# LANG = "uk"
DEVICE = "cpu"
TASK = "transcribe"

initial_prompt_data = None

if os.path.isfile("initial_prompt.json"):
    with open("initial_prompt.json", 'r', encoding='utf-8') as file:
        initial_prompt_json = json.load(file)
        print(initial_prompt_json)
    value = ""
    for key in initial_prompt_json:
        value = value + initial_prompt_json[key]

    initial_prompt_data = value

model = whisper.load_model(name=MODEL_NAME, device=DEVICE, in_memory=False)

print(f"using model: {MODEL_NAME} for {LANG}")

def sendToWhisper(audio_upload, results):
    start, result = datetime.now(), [None, None, None]

    if audio_upload is None:
        #return "ERROR: You have to upload an audio file"
        return [["No input"]*3]

    audio = audio_upload

    _, file_name = os.path.split(audio_upload)

    # Gradio inserts some 8 symbols before first dot symbol in original filename
    # e.g voice.mp3 -> voiceXXXXXXXX.mp3, voice1.2.mp3 -> voice1XXXXXXXX.2.mp3
    # Lets restore original filename
    dot_pos = file_name.index(".")
    file_name = file_name[:dot_pos-8] + file_name[dot_pos:]

    results.append(result)
    print(f"task started for: {file_name}")
    output = model.transcribe(audio, language=LANG, fp16=False, verbose=True, condition_on_previous_text=True, initial_prompt=initial_prompt_data)

    result[0], result[1], result[2] = file_name, output['text'], str((datetime.now() - start).total_seconds())
    print(f"task completed for: {file_name}")
    return results

CSS = """
#audio_inputs {
    height:100px;
    max-height:100px;
}
"""

with gr.Blocks(css=CSS) as demo:

    gr.Markdown("### [OpenAI Whisper](https://openai.com/blog/whisper/)  Demo")
    gr.Markdown("A [Gradio](https://gradio.app) based Speech-to-Text (aka ASR) Demo of the [Open AI Whisper Model](https://github.com/openai/whisper)")
    
    results = gr.State([])
    with gr.Column():

        gr.Markdown("### Upload audio")

        with gr.Row():
            audio_upload = gr.Audio(source="upload", type="filepath", interactive=True, elem_id="audio_inputs")

        with gr.Row():
            submit = gr.Button(value="Transcribe")

        gr.Markdown("### Result")
        output = gr.Dataframe(headers=["Filename", "Transcription", "Time [s]"], label="Results", wrap=True)

    submit.click(fn=sendToWhisper, inputs=[audio_upload, results], outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
