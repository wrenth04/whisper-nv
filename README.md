# whisper-nv

OpenAI-compatible transcription server backed by NVIDIA Riva offline ASR.

Uploaded audio is normalized to mono, 16-bit, little-endian PCM (`pcm_s16le`) at 16 kHz before it is sent to Riva.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

```bash
export RIVA_SERVER=localhost:50051
uvicorn openai_server:app --host 0.0.0.0 --port 8000
```

## Run against NVIDIA endpoint

```bash
export NVIDIA_API_KEY=your_api_key
export RIVA_SERVER=grpc.nvcf.nvidia.com:443
export RIVA_USE_SSL=true
export RIVA_METADATA="function-id:b702f636-f60c-4a3d-a6f4-f3568c13bd7d,authorization:Bearer $NVIDIA_API_KEY"

uvicorn openai_server:app --host 0.0.0.0 --port 8000
```

Supported upload formats depend on the bundled decoder library and do not use `ffmpeg`. Common formats such as WAV, MP3, FLAC, and OGG are expected to work. Each upload is decoded and resampled to mono 16-bit PCM at 16 kHz before transcription. Unsupported or unreadable formats return HTTP 400.

Optional environment variables:

- `OPENAI_API_KEY`: require `Authorization: Bearer ...`
- `RIVA_USE_SSL=true`
- `RIVA_SSL_ROOT_CERT=/path/to/ca.pem`
- `RIVA_SSL_CLIENT_CERT=/path/to/client.crt`
- `RIVA_SSL_CLIENT_KEY=/path/to/client.key`
- `RIVA_METADATA=key1:value1,key2:value2`
- `RIVA_MAX_MESSAGE_LENGTH=67108864`
- `ASR_VAD_MAX_SEGMENT_SECONDS=10` (VAD 切段最長秒數)
- `ASR_MAX_CHUNK_MB=50` (單次處理的 PCM 音訊上限，超過會分段呼叫 Riva 並合併時間軸)
- `ASR_NO_VERBATIM_TRANSCRIPTS=true` (優先在 decoder 端抑制口語重覆/贅字，降低重覆 token 輸出)

## Docker

Build the image:

```bash
docker build -t whisper-nv .
```

Run the container against the NVIDIA endpoint by setting only `NVIDIA_API_KEY`:

```bash
docker run --rm -p 8000:8000 \
  -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
  -e RIVA_MAX_MESSAGE_LENGTH=67108864 \
  whisper-nv
```

The container entrypoint will automatically set:

- `FUNCTION_ID=b702f636-f60c-4a3d-a6f4-f3568c13bd7d`
- `RIVA_SERVER=grpc.nvcf.nvidia.com:443`
- `RIVA_USE_SSL=true`
- `RIVA_METADATA=function-id:...,authorization:Bearer $NVIDIA_API_KEY`

You can still override `FUNCTION_ID`, `RIVA_SERVER`, `RIVA_USE_SSL`, or `RIVA_METADATA` explicitly if needed.

If you want this API server itself to require bearer auth, also pass:

```bash
-e OPENAI_API_KEY=$OPENAI_API_KEY
```

Only provide custom certificate paths when your Riva endpoint uses a private CA or requires mTLS. Otherwise you can omit them.

If you need custom certificates, mount them and point the env vars at the mounted paths:

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/certs:/certs:ro \
  -e RIVA_SERVER=grpc.nvcf.nvidia.com:443 \
  -e RIVA_USE_SSL=true \
  -e RIVA_SSL_ROOT_CERT=/certs/ca.pem \
  -e RIVA_SSL_CLIENT_CERT=/certs/client.crt \
  -e RIVA_SSL_CLIENT_KEY=/certs/client.key \
  -e RIVA_METADATA="function-id=b702f636-f60c-4a3d-a6f4-f3568c13bd7d,authorization=Bearer your_api_key" \
  whisper-nv
```

## Docker Compose

Copy the example file and provide `NVIDIA_API_KEY` in your shell or a `.env` file:

```bash
cp docker-compose.example.yml docker-compose.yml
export NVIDIA_API_KEY=your_api_key
docker compose up --build
```

The entrypoint will derive the NVIDIA endpoint settings from `NVIDIA_API_KEY` automatically. The example also raises `RIVA_MAX_MESSAGE_LENGTH` to 64 MB for larger uploads. Uncomment `OPENAI_API_KEY` if you want this API server to require bearer auth. Uncomment or override `FUNCTION_ID`, `RIVA_SERVER`, `RIVA_USE_SSL`, or the certificate settings only when you need non-default behavior.

## API

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/transcriptions`

`response_format=verbose_json` returns an OpenAI-style transcription shape with `text`, `duration`, `language`, `words`, and `segments`. Whisper-specific segment metadata fields that are not produced by Riva are returned with placeholder values.

### curl example

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F file=@sample.wav \
  -F model=whisper-1 \
  -F language=en \
  -F response_format=verbose_json
```
