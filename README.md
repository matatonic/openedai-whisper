OpenedAI Whisper
----------------

An OpenAI API compatible speech to text server for audio transcription and translations, aka. Whisper.

- Compatible with the OpenAI audio/transcriptions and audio/translations API
- Does not connect to the OpenAI API and does not require an OpenAI API Key
- Not affiliated with OpenAI in any way

API Compatibility:
- [X] /v1/audio/transcriptions
- [X] /v1/audio/translations

Parameter Support:
- [X] `file`
- [X] `model` (only whisper-1 exists, so this is ignored)
- [X] `language`
- [ ] `prompt` (not yet supported)
- [X] `temperature`
- [X] `response_format`:
- - [X] `json`
- - [X] `text`
- - [X] `srt`
- - [X] `vtt`
- - [X] `verbose_json` *(partial support, some fields missing)

Details:
* CUDA or CPU support (automatically detected)
* float32, float16 or bfloat16 support (automatically detected)

Tested whisper models:
* openai/whisper-large-v2 (the default)
* openai/whisper-large-v3
* distil-whisper/distil-medium.en
* openai/whisper-tiny.en
* ...


Version: 0.1.0, Last update: 2024-03-15


API Documentation
-----------------

## Usage

* [OpenAI Speech to text guide](https://platform.openai.com/docs/guides/speech-to-text)
* [OpenAI API Transcription Reference](https://platform.openai.com/docs/api-reference/audio/createTranscription)
* [OpenAI API Translation Reference](https://platform.openai.com/docs/api-reference/audio/createTranslation)


Installation instructions
-------------------------

You will need to install CUDA for your operating system if you want to use CUDA.

```shell
# Install the Python requirements
pip install -r requirements.txt
# install ffmpeg
sudo apt install ffmpeg
```

Usage
-----

```
Usage: whisper.py [-m <model_name>] [-d <device>] [-t <dtype>] [-P <port>] [-H <host>] [--preload]


Description:
OpenedAI Whisper API Server

Options:
-h, --help            Show this help message and exit.
-m MODEL, --model MODEL
                      The model to use for transcription.
                      Ex. distil-whisper/distil-medium.en (default: openai/whisper-large-v2)
-d DEVICE, --device DEVICE
                      Set the torch device for the model. Ex. cuda:1 (default: auto)
-t DTYPE, --dtype DTYPE
                      Set the torch data type for processing (float32, float16, bfloat16) (default: auto)
-P PORT, --port PORT  Server tcp port (default: 8000)
-H HOST, --host HOST  Host to listen on, Ex. 0.0.0.0 (default: localhost)
--preload             Preload model and exit. (default: False)
```

Sample API Usage
----------------

You can use it like this:

```shell
curl -s http://localhost:8000/v1/audio/transscriptions -H "Content-Type: multipart/form-data" -F model="whisper-1" -F file="@audio.mp3" -F response_format=text
```

Or just like this:

```shell
curl -s http://localhost:8000/v1/audio/transscriptions -F model="whisper-1" -F file="@audio.mp3"
```

Or like this example from the [OpenAI Speech to text guide Quickstart](https://platform.openai.com/docs/guides/speech-to-text/quickstart):

```python
from openai import OpenAI
client = OpenAI(api_key='sk-1111', base_url='http://localhost:8000/v1')

audio_file = open("/path/to/file/audio.mp3", "rb")
transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
print(transcription.text)
```

Docker support
--------------

You can run the server via docker like so:
```shell
docker compose build
docker compose up
```

Options can be set via `whisper.env`.
