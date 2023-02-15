#!/usr/bin/env python3

import gradio as gr
import whisper
from datetime import datetime

MODEL_NAME = "large-v2"
LANG = "ru"
DEVICE = "cpu"
TASK = "transcribe"

model = whisper.load_model(MODEL_NAME, DEVICE)
options = whisper.DecodingOptions(fp16 = False, task=TASK, language=LANG)

print(f"using model: {MODEL_NAME} for {LANG}")

def sendToWhisper(audio_upload, results):
    start, result = datetime.now(), [None, None]

    if audio_upload is None:
        #return "ERROR: You have to upload an audio file"
        return [["No input"]*5]

    audio = audio_upload
    
    #TODO: implement algorithm from Whisper app to transcribe files longer than 30seconds instead of pad_or_trim
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(whisper.load_audio(audio))).to(model.device)

    results.append(result)
    
    output_text = whisper.decode(model, mel, options)

    result[0], result[1] = output_text.text, str((datetime.now() - start).total_seconds())
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
        output = gr.Dataframe(headers=["Transcription", "Time [s]"], label="Results", wrap=True)

    submit.click(fn=sendToWhisper, inputs=[audio_upload, results], outputs=output)

demo.launch(share=False)
