#!/usr/bin/env python3
import os
import io
import gradio as gr
import openai
from datetime import datetime
from subprocess import CalledProcessError, run
import json
import re

MODEL_NAME = "whisper-1"
LANG = "ru"
# LANG = "uk"

initial_prompt_data = None
replacement_data = None

if os.path.isfile("initial_prompt.json"):
    with open("initial_prompt.json", 'r', encoding='utf-8') as file:
        initial_prompt_json = json.load(file)
        print(initial_prompt_json)
    value = ""
    for key in initial_prompt_json:
        value = value + initial_prompt_json[key]

    initial_prompt_data = value

if os.path.isfile("replacements.json"):
    with open("replacements.json", 'r', encoding='utf-8') as file:
        replacement_data = json.load(file)


def replaceText(transcribtion_text):
    for key in replacement_data.keys():
        transcribtion_text = re.sub(
            r'(?i)'+key, replacement_data[key], transcribtion_text)

    return transcribtion_text


def convertAudio(audio):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", audio,
        "-ac", "1",
        "-acodec", "libmp3lame",
        "-f", "mp3",
        "pipe:1"
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return out


def sendToWhisper(audio_upload, results):
    start, result = datetime.now(), [None, None, None]

    if audio_upload is None:
        # return "ERROR: You have to upload an audio file"
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

    # openai.Audio supports m4a, mp3, webm, mp4, mpga, wav, mpeg formats and it looks first format check
    # is just file extension check. So let just name file as "audio.mp3" and converts all audio data to mp3
    audio_data = io.BytesIO(convertAudio(audio))
    audio_data.name = "audio.mp3"
    transcription = openai.Audio.transcribe(
        MODEL_NAME, audio_data, language=LANG, prompt=initial_prompt_data)

    if replacement_data:
        text = replaceText(transcription["text"])
    else:
        text = transcription["text"]

    result[0], result[1], result[2] = file_name, text, str(
        (datetime.now() - start).total_seconds())
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
    gr.Markdown(
        "A [Gradio](https://gradio.app) based Speech-to-Text (aka ASR) Demo of the [Open AI API](https://github.com/openai/openai-python)")

    results = gr.State([])
    with gr.Column():

        gr.Markdown("### Upload audio")

        with gr.Row():
            # Gradio supports formats for player
            audio_upload = gr.Audio(
                source="upload", type="filepath", interactive=True, elem_id="audio_inputs")

        with gr.Row():
            submit = gr.Button(value="Transcribe")

        gr.Markdown("### Result")
        output = gr.Dataframe(
            headers=["Filename", "Transcription", "Time [s]"], label="Results", wrap=True)

    submit.click(fn=sendToWhisper, inputs=[
                 audio_upload, results], outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
