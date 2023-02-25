#!/usr/bin/env python3
import os
import gradio as gr
from datetime import datetime
from faster_whisper import WhisperModel
import json

# MODEL_NAME = "whisper-large-v2-ct2-int8/"
MODEL_NAME = "whisper-large-v2-ct2-f16/"
""" mitchelldehaven/whisper-large-v2-ru """
# MODEL_NAME = "whisper-large-v2-ru-f16/"
# MODEL_NAME = "whisper-large-v2-ru-int8/"
""" VadymHromov/whisper-small-ru """
# MODEL_NAME = "whisper-small-ru-f16/"
# MODEL_NAME = "whisper-small-ru-int8/"
LANG = "ru"
DEVICE = "cpu"

initial_prompt_data = None

if os.path.isfile("initial_prompt.json"):
    with open("initial_prompt.json", 'r', encoding='utf-8') as file:
        initial_prompt_json = json.load(file)
        print(initial_prompt_json)
    value = ""
    for key in initial_prompt_json:
        value = value + initial_prompt_json[key]

    initial_prompt_data = value

model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="float32", cpu_threads=4)
# model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8", cpu_threads=4)

print(f"using model: {MODEL_NAME} for {LANG}")

def sendToWhisper(audio_upload, results):
    output_text = ""
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
    segments, _ = model.transcribe(input_file=audio, beam_size=5, language=LANG, without_timestamps=False,  initial_prompt=initial_prompt_data, temperature=0.0)

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        output_text = output_text + segment.text

    result[0], result[1], result[2] = file_name, output_text, str((datetime.now() - start).total_seconds())
    print(f"task completed for: {file_name}")
    return results

CSS = """
#audio_inputs {
    height:100px;
    max-height:100px;
}
"""

with gr.Blocks(css=CSS) as demo:

    gr.Markdown("[Faster Whisper](https://github.com/guillaumekln/faster-whisper) Demo")
    gr.Markdown("A [Gradio](https://gradio.app) based Speech-to-Text (aka ASR) Demo of the [Open AI Whisper Model](https://github.com/openai/whisper)")
    
    results = gr.State([])
    with gr.Column():

        with gr.Row():
            audio_upload = gr.Audio(source="upload", type="filepath", interactive=True, elem_id="audio_inputs", label="Upload audio")

        with gr.Row():
            submit = gr.Button(value="Transcribe")

        output = gr.Dataframe(headers=["Filename", "Transcription", "Time [s]"], label="Results", wrap=True)

    submit.click(fn=sendToWhisper, inputs=[audio_upload, results], outputs=output)

demo.launch(share=False)
