from __future__ import annotations

from dataclasses import dataclass

import miniaudio


TARGET_SAMPLE_RATE_HZ = 16000


class AudioDecodeError(ValueError):
    pass


@dataclass
class DecodedAudio:
    pcm_bytes: bytes
    sample_rate_hz: int


def decode_to_pcm_s16le(
    audio_bytes: bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
) -> DecodedAudio:
    try:
        decoded = miniaudio.decode(
            audio_bytes,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=TARGET_SAMPLE_RATE_HZ,
        )
    except miniaudio.DecodeError as exc:
        detail = _describe_source(filename, content_type)
        raise AudioDecodeError(f"Unsupported or unreadable audio format{detail}") from exc

    sample_rate_hz = int(getattr(decoded, "sample_rate", 0) or 0)
    pcm_bytes = bytes(getattr(decoded, "samples", b""))
    if sample_rate_hz <= 0 or not pcm_bytes:
        detail = _describe_source(filename, content_type)
        raise AudioDecodeError(f"Decoded audio is empty or missing sample rate{detail}")

    return DecodedAudio(pcm_bytes=pcm_bytes, sample_rate_hz=sample_rate_hz)


def _describe_source(filename: str | None, content_type: str | None) -> str:
    parts: list[str] = []
    if filename:
        parts.append(f"filename={filename}")
    if content_type:
        parts.append(f"content_type={content_type}")
    if not parts:
        return ""
    return f" ({', '.join(parts)})"
