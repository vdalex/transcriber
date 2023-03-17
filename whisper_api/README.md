Install requirements:

``` bash
pip install -r requirements.txt
```

On Windows PowerShell:

``` bash
$Env:OPENAI_API_KEY = "openai_api_key"
python whisper_demo_api.py
```

Output example:
```bash
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
task started for: audio_sample.mp3
task completed for: audio_sample.mp3
```

Web GUI:

![interface](../assets/web_gui_1.png)

![file_uploaded](../assets/web_gui_2.png)

![file_transcribed](../assets/web_gui_3.png)

replacements.json example:

```json
{
    "word":"word_1"
}
```

To check API usage

https://platform.openai.com/account/usage
