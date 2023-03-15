To convert hugging face models to ct2 format:

```bash
ct2-transformers-converter.exe --model openai/whisper-large-v2 --output_dir models/whisper-large-v2-ct2-f16 --quantization float16
```