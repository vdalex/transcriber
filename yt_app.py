from pytube import YouTube
from transformers import WhisperForConditionalGeneration
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor


mymodel = "openai/whisper-medium"
#mymodel = "openai/whisper-large"
#mymodel = "emilios/whisper-medium-el"
#lang="English"
lang="Greek"

model = WhisperForConditionalGeneration.from_pretrained( mymodel)

tokenizer = WhisperTokenizer.from_pretrained( mymodel, language=lang, task="transcribe")

processor = WhisperProcessor.from_pretrained( mymodel, language=lang, task="transcribe")

feature_extractor = WhisperFeatureExtractor.from_pretrained( mymodel, language=lang, task="transcribe")

link = 'https://www.youtube.com/watch?v=e_eCryyPRus'

try:
 yt = YouTube(link)
except:
 print("Connection Error")

yt.streams.filter(file_extension='mp4')
stream = yt.streams.get_by_itag(139)
stream.download('',"YouTube.mp4")

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "el", task = "transcribe")
model.config.suppress_tokens = []
#model.config.max_new_tokens = 1024

from transformers import pipeline

transcript = pipeline(
    task="automatic-speech-recognition",
    model = model,
    feature_extractor = feature_extractor,
    tokenizer=tokenizer,
    framework="pt",
    batch_size=16,
    device='cuda:0',
    #generate_kwargs={"max_new_tokens": 1024},
    #max_new_tokens = 1024,
    chunk_length_s=30, # 12
    stride_length_s=(5, 5), # must have with chunk_length_s
    condition_on_previous_text=0,
    compression_ratio_threshold=2.4
)

out = transcript(["YouTube.mp4"])
print(out)