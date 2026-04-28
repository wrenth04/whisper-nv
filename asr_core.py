from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import grpc
import riva.client


@dataclass
class ASRConfigOptions:
    language_code: str | None = None
    model_name: str | None = None
    max_alternatives: int = 1
    profanity_filter: bool = False
    automatic_punctuation: bool = True
    no_verbatim_transcripts: bool = False
    word_time_offsets: bool = False
    speaker_diarization: bool = False
    diarization_max_speakers: int = 8
    boosted_lm_words: list[str] | None = None
    boosted_lm_score: float = 4.0
    start_history: int = -1
    start_threshold: float = -1.0
    stop_history: int = -1
    stop_history_eou: int = -1
    stop_threshold: float = -1.0
    stop_threshold_eou: float = -1.0
    custom_configuration: str = ""


@dataclass
class WordTimestamp:
    word: str
    start: float | None
    end: float | None
    speaker: str | None = None


@dataclass
class Segment:
    id: int
    start: float | None
    end: float | None
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    duration: float | None
    words: list[WordTimestamp]
    segments: list[Segment]
    raw_response: Any


class RivaTranscriptionError(RuntimeError):
    pass


class OfflineASRClient:
    def __init__(
        self,
        *,
        server: str,
        use_ssl: bool = False,
        ssl_root_cert: str | None = None,
        ssl_client_cert: str | None = None,
        ssl_client_key: str | None = None,
        metadata: list[str] | None = None,
        max_message_length: int = 16 * 1024 * 1024,
    ) -> None:
        options = [
            ("grpc.max_receive_message_length", max_message_length),
            ("grpc.max_send_message_length", max_message_length),
        ]
        auth = riva.client.Auth(
            ssl_root_cert=ssl_root_cert,
            ssl_client_cert=ssl_client_cert,
            ssl_client_key=ssl_client_key,
            use_ssl=use_ssl,
            uri=server,
            metadata_args=metadata,
            options=options,
        )
        self.asr_service = riva.client.ASRService(auth)

    def list_models(self) -> dict[str, list[dict[str, list[str]]]]:
        config_response = self.asr_service.stub.GetRivaSpeechRecognitionConfig(
            riva.client.proto.riva_asr_pb2.RivaSpeechRecognitionConfigRequest()
        )
        asr_models: dict[str, list[dict[str, list[str]]]] = {}
        for model_config in config_response.model_config:
            if model_config.parameters.get("type") != "offline":
                continue
            language_code = model_config.parameters.get("language_code", "")
            model = {"model": [model_config.model_name]}
            asr_models.setdefault(language_code, []).append(model)
        return dict(sorted(asr_models.items()))

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate_hz: int,
        config_options: ASRConfigOptions,
    ) -> TranscriptionResult:
        config = riva.client.RecognitionConfig(
            encoding=1,
            sample_rate_hertz=sample_rate_hz,
            language_code=config_options.language_code or "",
            model=config_options.model_name or "",
            max_alternatives=config_options.max_alternatives,
            profanity_filter=config_options.profanity_filter,
            enable_automatic_punctuation=config_options.automatic_punctuation,
            verbatim_transcripts=not config_options.no_verbatim_transcripts,
            enable_word_time_offsets=config_options.word_time_offsets or config_options.speaker_diarization,
        )
        riva.client.add_word_boosting_to_config(config, config_options.boosted_lm_words, config_options.boosted_lm_score)
        riva.client.add_speaker_diarization_to_config(
            config,
            config_options.speaker_diarization,
            config_options.diarization_max_speakers,
        )
        riva.client.add_endpoint_parameters_to_config(
            config,
            config_options.start_history,
            config_options.start_threshold,
            config_options.stop_history,
            config_options.stop_history_eou,
            config_options.stop_threshold,
            config_options.stop_threshold_eou,
        )
        riva.client.add_custom_configuration_to_config(config, config_options.custom_configuration)

        try:
            response = self.asr_service.offline_recognize(audio_bytes, config)
        except grpc.RpcError as exc:
            raise RivaTranscriptionError(exc.details()) from exc

        words: list[WordTimestamp] = []
        segments: list[Segment] = []
        text_parts: list[str] = []

        previous_segment_end = 0.0
        for index, result in enumerate(getattr(response, "results", [])):
            if not getattr(result, "alternatives", None):
                continue
            alternative = result.alternatives[0]
            transcript = alternative.transcript.strip()
            if transcript:
                text_parts.append(transcript)

            segment_words: list[WordTimestamp] = []
            for word_info in getattr(alternative, "words", []):
                word = WordTimestamp(
                    word=getattr(word_info, "word", ""),
                    start=_normalize_time(getattr(word_info, "start_time", None)),
                    end=_normalize_time(getattr(word_info, "end_time", None)),
                    speaker=_speaker_label(getattr(word_info, "speaker_tag", None)),
                )
                words.append(word)
                segment_words.append(word)

            processed_end = _normalize_audio_processed(getattr(result, "audio_processed", None))
            word_start = segment_words[0].start if segment_words else None
            word_end = segment_words[-1].end if segment_words else None

            segment_start = previous_segment_end
            if word_start is not None:
                segment_start = max(previous_segment_end, min(word_start, processed_end if processed_end is not None else word_start))

            segment_end = processed_end
            if segment_end is None:
                segment_end = word_end
            elif word_end is not None:
                segment_end = max(segment_start, max(word_end, processed_end))

            if segment_end is None:
                segment_end = segment_start

            segments.append(
                Segment(
                    id=index,
                    start=segment_start,
                    end=segment_end,
                    text=transcript,
                )
            )
            previous_segment_end = segment_end

        duration = None
        if words and words[-1].end is not None:
            duration = words[-1].end
        elif segments and segments[-1].end is not None:
            duration = segments[-1].end

        return TranscriptionResult(
            text=" ".join(part for part in text_parts if part).strip(),
            language=config_options.language_code,
            duration=duration,
            words=words,
            segments=segments,
            raw_response=response,
        )


def _speaker_label(speaker_tag: Any) -> str | None:
    if speaker_tag in (None, 0, "0"):
        return None
    try:
        return f"speaker{int(speaker_tag)}"
    except (TypeError, ValueError):
        return None


def _normalize_time(value: Any) -> float | None:
    if value is None:
        return None

    seconds = getattr(value, "seconds", None)
    nanos = getattr(value, "nanos", None)
    if seconds is not None:
        return float(seconds) + float(nanos or 0) / 1_000_000_000

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    return numeric / 1000.0


def _normalize_audio_processed(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None
