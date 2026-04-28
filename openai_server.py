from __future__ import annotations

import os
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from asr_core import (
    ASRConfigOptions,
    OfflineASRClient,
    RivaTranscriptionError,
    merge_transcription_results,
    split_pcm_by_size,
)
from audio_decode import AudioDecodeError, decode_to_pcm_s16le


app = FastAPI(title="whisper-nv", version="0.1.0")
SUPPORTED_VAD_ENGINES = {"auto", "webrtc", "silero", "energy"}


def get_client() -> OfflineASRClient:
    metadata = os.getenv("RIVA_METADATA")
    metadata_args = None
    if metadata:
        metadata_args = []
        for item in metadata.split(","):
            entry = item.strip()
            if not entry:
                continue
            key, sep, value = entry.partition(":")
            if not sep:
                raise ValueError("RIVA_METADATA entries must use key:value format")
            metadata_args.append([key.strip(), value.strip()])
    return OfflineASRClient(
        server=os.getenv("RIVA_SERVER", "localhost:50051"),
        use_ssl=os.getenv("RIVA_USE_SSL", "false").lower() == "true",
        ssl_root_cert=os.getenv("RIVA_SSL_ROOT_CERT"),
        ssl_client_cert=os.getenv("RIVA_SSL_CLIENT_CERT"),
        ssl_client_key=os.getenv("RIVA_SSL_CLIENT_KEY"),
        metadata=metadata_args,
        max_message_length=int(os.getenv("RIVA_MAX_MESSAGE_LENGTH", str(16 * 1024 * 1024))),
    )


client = get_client()
api_key = os.getenv("OPENAI_API_KEY")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models(authorization: Annotated[str | None, Header()] = None) -> dict[str, object]:
    _check_auth(authorization)
    models = client.list_models()
    data = []
    for language, items in models.items():
        for item in items:
            model_name = item["model"][0]
            data.append(
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nvidia-riva",
                    "language": language,
                }
            )
    return {"object": "list", "data": data}


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    timestamp_granularities: Annotated[list[str] | None, Form()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    del prompt
    _check_auth(authorization)

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        decoded_audio = decode_to_pcm_s16le(
            audio_bytes,
            filename=file.filename,
            content_type=file.content_type,
        )
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    wants_verbose = response_format == "verbose_json"
    config_options = ASRConfigOptions(
        language_code=language,
        model_name=None if model == "whisper-1" else model,
        word_time_offsets=wants_verbose or _wants_word_timestamps(timestamp_granularities),
    )

    max_chunk_mb = float(os.getenv("ASR_MAX_CHUNK_MB", "50"))
    max_chunk_bytes = int(max_chunk_mb * 1024 * 1024)
    max_vad_segment_seconds = float(os.getenv("ASR_VAD_MAX_SEGMENT_SECONDS", "10"))
    vad_engine = _get_vad_engine_from_env()

    chunks, offsets = split_pcm_by_size(decoded_audio.pcm_bytes, decoded_audio.sample_rate_hz, max_chunk_bytes)

    try:
        partial_results = [
            client.transcribe_vad_packed(
                chunk,
                decoded_audio.sample_rate_hz,
                config_options,
                max_vad_segment_seconds=max_vad_segment_seconds,
                max_request_bytes=max_chunk_bytes,
                vad_engine=vad_engine,
            )
            for chunk in chunks
        ]
        result = merge_transcription_results(partial_results, offsets)
    except RivaTranscriptionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if response_format == "text":
        return PlainTextResponse(result.text)
    if response_format == "json":
        return JSONResponse({"text": result.text})
    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": "transcribe",
                "language": result.language or language,
                "duration": result.duration,
                "text": result.text,
                "words": [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                    }
                    for word in result.words
                ],
                "segments": [
                    {
                        "id": segment.id,
                        "seek": int((segment.start or 0.0) * 100),
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "tokens": [],
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                    }
                    for segment in result.segments
                ],
            }
        )
    if response_format == "srt":
        return PlainTextResponse(_to_srt(result.segments), media_type="text/plain; charset=utf-8")
    if response_format == "vtt":
        return PlainTextResponse(_to_vtt(result.segments), media_type="text/vtt; charset=utf-8")

    raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")


def _check_auth(authorization: str | None) -> None:
    if not api_key:
        return
    if authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _wants_word_timestamps(timestamp_granularities: list[str] | None) -> bool:
    if not timestamp_granularities:
        return False
    return "word" in timestamp_granularities


def _get_vad_engine_from_env() -> str:
    configured = os.getenv("ASR_VAD_ENGINE", "auto").strip().lower()
    if configured in SUPPORTED_VAD_ENGINES:
        return configured
    return "auto"


def _to_srt(segments) -> str:
    lines: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        if segment.start is None or segment.end is None or not segment.text:
            continue
        lines.append(str(idx))
        lines.append(f"{_format_srt_time(segment.start)} --> {_format_srt_time(segment.end)}")
        lines.append(segment.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _to_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        if segment.start is None or segment.end is None or not segment.text:
            continue
        lines.append(f"{_format_vtt_time(segment.start)} --> {_format_vtt_time(segment.end)}")
        lines.append(segment.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _format_srt_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    whole_seconds = int(remainder)
    milliseconds = int(round((remainder - whole_seconds) * 1000))
    return f"{int(hours):02}:{int(minutes):02}:{whole_seconds:02},{milliseconds:03}"


def _format_vtt_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    whole_seconds = int(remainder)
    milliseconds = int(round((remainder - whole_seconds) * 1000))
    return f"{int(hours):02}:{int(minutes):02}:{whole_seconds:02}.{milliseconds:03}"
