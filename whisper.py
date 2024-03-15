#!/usr/bin/env python3
import os
import sys
import argparse

import torch
from transformers import pipeline
from typing import Optional, List
from fastapi import UploadFile, Form
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

import openedai

pipe = None
app = openedai.OpenAIStub()

async def whisper(file, response_format: str, **kwargs):
    global pipe

    result = pipe(await file.read(), **kwargs)

    filename_noext, ext = os.path.splitext(file.filename)

    if response_format == "text":
        return PlainTextResponse(result["text"].strip(), headers={"Content-Disposition": f"attachment; filename={filename_noext}.txt"})

    elif response_format == "json":
        return JSONResponse(content={ 'text': result['text'].strip() }, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}.json"})
    
    elif response_format == "verbose_json":
        chunks = result["chunks"]

        response = {
            "task": kwargs['generate_kwargs']['task'],
            #"language": "english",
            "duration": chunks[-1]['timestamp'][1],
            "text": result["text"].strip(),
        }
        if kwargs['return_timestamps'] == 'word':
            response['words'] = [{'word': chunk['text'].strip(), 'start': chunk['timestamp'][0], 'end': chunk['timestamp'][1] } for chunk in chunks ]
        else:
            response['segments'] = [{
                    "id": i,
                    #"seek": 0,
                    'start': chunk['timestamp'][0],
                    'end': chunk['timestamp'][1],
                    'text': chunk['text'].strip(),
                    #"tokens": [ ],
                    #"temperature": 0.0,
                    #"avg_logprob": -0.2860786020755768,
                    #"compression_ratio": 1.2363636493682861,
                    #"no_speech_prob": 0.00985979475080967
            } for i, chunk in enumerate(chunks) ]
        
        return JSONResponse(content=response, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={filename_noext}_verbose.json"})

    elif response_format == "srt":
            def srt_time(t):
                return "{:02d}:{:02d}:{:06.3f}".format(int(t//3600), int(t//60)%60, t%60).replace(".", ",")

            return PlainTextResponse("\n".join([ f"{i}\n{srt_time(chunk['timestamp'][0])} --> {srt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                for i, chunk in enumerate(result["chunks"], 1) ]), media_type="text/srt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.srt"})

    elif response_format == "vtt":
            def vtt_time(t):
                return "{:02d}:{:06.3f}".format(int(t//60), t%60)
            
            return PlainTextResponse("\n".join(["WEBVTT\n"] + [ f"{vtt_time(chunk['timestamp'][0])} --> {vtt_time(chunk['timestamp'][1])}\n{chunk['text'].strip()}\n"
                for chunk in result["chunks"] ]), media_type="text/vtt; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename_noext}.vtt"})


@app.post("/v1/audio/transcriptions")
async def transcriptions(
        file: UploadFile,
        model: str = Form(...),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
        timestamp_granularities: List[str] = Form(["segment"])
    ):
    global pipe

    kwargs = {'generate_kwargs': {'task': 'transcribe'}}

    if language:
        kwargs['generate_kwargs']["language"] = language
# May work soon, https://github.com/huggingface/transformers/issues/27317
#    if prompt:
#        kwargs["initial_prompt"] = prompt
    if temperature:
        kwargs['generate_kwargs']["temperature"] = temperature
        kwargs['generate_kwargs']['do_sample'] = True

    if response_format == "verbose_json" and 'word' in timestamp_granularities:
        kwargs['return_timestamps'] = 'word'
    else:
        kwargs['return_timestamps'] = response_format in ["verbose_json", "srt", "vtt"]

    return await whisper(file, response_format, **kwargs)


@app.post("/v1/audio/translations")
async def translations(
        file: UploadFile,
        model: str = Form(...),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
        temperature: Optional[float] = Form(None),
    ):
    global pipe

    kwargs = {'generate_kwargs': {"task": "translate"}}

# May work soon, https://github.com/huggingface/transformers/issues/27317
#    if prompt:
#        kwargs["initial_prompt"] = prompt
    if temperature:
        kwargs['generate_kwargs']["temperature"] = temperature
        kwargs['generate_kwargs']['do_sample'] = True

    kwargs['return_timestamps'] = response_format in ["verbose_json", "srt", "vtt"]

    return await whisper(file, response_format, **kwargs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog='whisper.py',
        description='OpenedAI Whisper API Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model', action='store', default="openai/whisper-large-v2", help="The model to use for transcription. Ex. distil-whisper/distil-medium.en")
    parser.add_argument('-d', '--device', action='store', default="auto", help="Set the torch device for the model. Ex. cuda:1")
    parser.add_argument('-t', '--dtype', action='store', default="auto", help="Set the torch data type for processing (float32, float16, bfloat16)")
    parser.add_argument('-P', '--port', action='store', default=8000, type=int, help="Server tcp port")
    parser.add_argument('-H', '--host', action='store', default='localhost', help="Host to listen on, Ex. 0.0.0.0")
    parser.add_argument('--preload', action='store_true', help="Preload model and exit.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dtype == "auto":
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 if args.dtype == "float16" else torch.float32

        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("bfloat16 not supported on this hardware, falling back to float16", file=sys.stderr)
            dtype = torch.float16

    pipe = pipeline("automatic-speech-recognition", model=args.model, device=device, chunk_length_s=30, torch_dtype=dtype)
    if args.preload:
        sys.exit(0)

    app.register_model('whisper-1', args.model)

    uvicorn.run(app, host=args.host, port=args.port) # , root_path=cwd, access_log=False, log_level="info", ssl_keyfile="cert.pem", ssl_certfile="cert.pem")
