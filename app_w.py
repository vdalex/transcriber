import torch
import gradio as gr

from transformers import WhisperForConditionalGeneration
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor
from transformers import pipeline

MODEL_NAME = "openai/whisper-large-v2"
# MODEL_NAME = "openai/whisper-medium"
# MODEL_NAME = "mitchelldehaven/whisper-large-v2-ru"
lang = "ru"

device = 0 if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_NAME, language=lang, task="transcribe")
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language=lang, task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    MODEL_NAME, language=lang, task="transcribe")

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=lang, task="transcribe")
model.config.suppress_tokens = []

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    framework="pt",
    batch_size=16,
    chunk_length_s=30,
    stride_length_s=(5, 5),
    condition_on_previous_text=0,
    compression_ratio_threshold=2.4,
    device="cpu",
)


def transcribe(file_upload):
    warn_output = ""
    if file_upload is None:
        return "ERROR: You have to upload an audio file"

    file = file_upload
    text = pipe(file)["text"]

    return warn_output + text


demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="upload", type="filepath", optional=True),
    ],
    outputs="text",
    layout="horizontal",
    theme="huggingface",
    # title="Transcribe Audio",
    description=(
        "Server uses"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe], ["Transcribe Audio"])

demo.launch(enable_queue=True, share=False)
