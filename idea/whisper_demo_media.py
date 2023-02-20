#!/usr/bin/env python3
import gradio as gr
import whisper
from datetime import datetime

MODEL_NAME = "large-v2"
#MODEL_NAME = "large-v1"
LANG = "ru"
DEVICE = "cpu"
TASK = "transcribe"
SAMPLE_RATE = 16000

model = whisper.load_model(name=MODEL_NAME, device=DEVICE, in_memory=False)

print(f"using model: {MODEL_NAME} for {LANG}")

def sendToWhisper(file_upload: gr.components.File, results):
    start, result = datetime.now(), [None, None, None]

    if file_upload is None:
        return [["No input"]*3]

    results.append(result)
    output = model.transcribe(file_upload.name, language=LANG, fp16=False, verbose=True, condition_on_previous_text=True, beam_size=3)

    result[0], result[1], result[2] = file_upload.orig_name, output['text'], str((datetime.now() - start).total_seconds())
    return results

def getAudioData(file_upload: gr.components.File):
    audio = whisper.load_audio(file_upload.name);
    return SAMPLE_RATE, audio

CSS = """
#audio_outputs {
    height:100px;
    max-height:100px;
}
"""

with gr.Blocks(css=CSS) as demo:

    gr.Markdown("### A [Gradio](https://gradio.app) based Speech-to-Text (aka ASR) Demo of the [Open AI Whisper Model](https://github.com/openai/whisper)")
    
    results = gr.State([])
    with gr.Column():

        gr.Markdown("### Upload audio")

        with gr.Row():
            file_upload = gr.File(interactive=True, elem_id="file_inputs", type="file", file_types=["audio", "video"])
        with gr.Row():
            audio_output = gr.Audio(elem_id="audio_outputs")
        with gr.Row():
            submit = gr.Button(value="Transcribe")

        gr.Markdown("### Result")
        output = gr.Dataframe(headers=["Filename", "Transcription", "Time [s]"], label="Results", wrap=True)

    submit.click(fn=sendToWhisper, inputs=[file_upload, results], outputs=output)
    file_upload.change(fn=getAudioData, inputs=[file_upload], outputs=audio_output)

demo.launch(share=True)
