#!/usr/bin/env python3

import gradio as gr
import whisper
from datetime import datetime

def sendToWhisper(audio_record, audio_upload, task, model_name, language_selected, results):
    start, result = datetime.now(), [model_name, None, language_selected, None, None]

    audio = audio_record or audio_upload
    if audio is None:
        return [["No input"]*5]

    model = whisper.load_model(model_name, "cpu")
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(whisper.load_audio(audio))).to(model.device)

    results.append(result)

    prob, language = 0, language_selected

    if language == "none":
        langugage = None
    elif language == "detect":
        _, probs = model.detect_language(mel)
        language = max(probs, key=probs.get)
        prob = probs[language]
        print(f"detected language: {model_name} - {language}")

    result[2:4] = [language, str(prob)]
    print(f"using model: {model_name} - {language} - {task}")

    options = whisper.DecodingOptions(fp16 = False, task=task, language=language)
    output_text = whisper.decode(model, mel, options)

    result[1], result[4] = output_text.text, str((datetime.now() - start).total_seconds())
    return results

CSS = """
#audio_inputs {
    height:100px;
    max-height:100px;
}
"""
LANGUAGES = [ "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "iw", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su" ]

all_models = whisper.available_models()
with gr.Blocks(css=CSS) as demo:

    gr.Markdown("### [OpenAI Whisper](https://openai.com/blog/whisper/)  Demo")
    gr.Markdown("A [Gradio](https://gradio.app) based Speech-to-Text (aka ASR) Demo of the [Open AI Whisper Model](https://github.com/openai/whisper)")
    gr.Markdown('<p align="right"><a href="https://huggingface.co/spaces/davidtsong/whisper-demo">inspired</a> and built like <a href="https://gist.github.com/kpe/6a70395ce171ffee43d927eaf90b81b6/#file-openai-whisper-demo-md">so</a></p>')

    results = gr.State([])
    with gr.Column():

        gr.Markdown("### Record or upload audio")

        with gr.Row():
            audio_record = gr.Audio(source="microphone", type="filepath", elem_id="audio_inputs", label="Recorded Audio")
            audio_upload = gr.Audio(source="upload", type="filepath", interactive=True, elem_id="audio_inputs")

        gr.Markdown("### Select a model and language")

        with gr.Row():
            models_selected = gr.Dropdown(all_models, label="Whisper model to use", value="tiny")
            language = gr.Dropdown(["detect", "none"] + LANGUAGES, label="Language", value="ru")
            task = gr.Dropdown(["transcribe", "translate"], label="Task", value="transcribe")
        submit = gr.Button(value="Transcribe")

        gr.Markdown("### Result")
        output = gr.Dataframe(headers=["Model", "Transcription", "Language", "Language Confidence", "Time [s]"], label="Results", wrap=True)

    submit.click(fn=sendToWhisper, inputs=[audio_record, audio_upload, task, models_selected, language, results], outputs=output)

demo.launch(share=False)
