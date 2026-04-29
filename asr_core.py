from __future__ import annotations

from array import array
from dataclasses import dataclass
import importlib
import importlib.util
import re
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
    word_time_offsets: bool = True
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


@dataclass
class PostProcessOptions:
    dedupe_repeated_phrases: bool = True
    max_consecutive_phrase_repeats: int = 3
    dedupe_adjacent_segments: bool = True
    adjacent_segment_similarity_threshold: float = 0.96
    collapse_music_tokens: bool = True
    max_consecutive_char_run: int = 12


@dataclass
class VADChunk:
    start_sample: int
    end_sample: int


@dataclass
class PackedPlacement:
    original_start: float
    packed_start: float
    duration: float
    slot_duration: float


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

        return _response_to_transcription(response, config_options.language_code)

    def transcribe_vad_packed(
        self,
        audio_bytes: bytes,
        sample_rate_hz: int,
        config_options: ASRConfigOptions,
        *,
        max_vad_segment_seconds: float = 10.0,
        max_request_bytes: int = 50 * 1024 * 1024,
    ) -> TranscriptionResult:
        segments = _vad_segments(audio_bytes, sample_rate_hz, max_vad_segment_seconds=max_vad_segment_seconds)
        mapped_words: list[WordTimestamp] = []
        mapped_segments: list[Segment] = []
        segment_id = 0

        for packed_audio, placements in _build_packed_audio_batches(
            audio_bytes,
            sample_rate_hz,
            segments,
            max_request_bytes=max_request_bytes,
        ):
            packed_result = self.transcribe_bytes(packed_audio, sample_rate_hz, config_options)

            for word in packed_result.words:
                mapped = _map_word(word, placements)
                if mapped:
                    mapped_words.append(mapped)

            for seg in packed_result.segments:
                mapped = _map_segment(seg, placements)
                if not mapped or not mapped.text:
                    continue
                mapped.id = segment_id
                mapped_segments.append(mapped)
                segment_id += 1

        mapped_words.sort(key=lambda w: ((w.start or 0.0), (w.end or w.start or 0.0)))
        mapped_segments.sort(key=lambda s: ((s.start or 0.0), (s.end or s.start or 0.0)))

        duration = None
        if mapped_words and mapped_words[-1].end is not None:
            duration = mapped_words[-1].end
        elif mapped_segments and mapped_segments[-1].end is not None:
            duration = mapped_segments[-1].end

        return TranscriptionResult(
            text=" ".join(seg.text for seg in mapped_segments if seg.text).strip(),
            language=config_options.language_code,
            duration=duration,
            words=mapped_words,
            segments=mapped_segments,
            raw_response={"mode": "vad_packed", "segment_count": len(segments)},
        )


def _response_to_transcription(response: Any, language_code: str | None) -> TranscriptionResult:
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
        language=language_code,
        duration=duration,
        words=words,
        segments=segments,
        raw_response=response,
    )


def merge_transcription_results(results: list[TranscriptionResult], offsets_seconds: list[float]) -> TranscriptionResult:
    merged_words: list[WordTimestamp] = []
    merged_segments: list[Segment] = []
    next_id = 0

    for result, offset in zip(results, offsets_seconds, strict=False):
        for word in result.words:
            merged_words.append(
                WordTimestamp(
                    word=word.word,
                    start=None if word.start is None else word.start + offset,
                    end=None if word.end is None else word.end + offset,
                    speaker=word.speaker,
                )
            )
        for segment in result.segments:
            merged_segments.append(
                Segment(
                    id=next_id,
                    start=None if segment.start is None else segment.start + offset,
                    end=None if segment.end is None else segment.end + offset,
                    text=segment.text,
                )
            )
            next_id += 1

    merged_words.sort(key=lambda w: ((w.start or 0.0), (w.end or w.start or 0.0)))
    merged_segments.sort(key=lambda s: ((s.start or 0.0), (s.end or s.start or 0.0)))

    duration = None
    if merged_words and merged_words[-1].end is not None:
        duration = merged_words[-1].end
    elif merged_segments and merged_segments[-1].end is not None:
        duration = merged_segments[-1].end

    return TranscriptionResult(
        text=" ".join(segment.text for segment in merged_segments if segment.text).strip(),
        language=results[0].language if results else None,
        duration=duration,
        words=merged_words,
        segments=merged_segments,
        raw_response={"merged_results": len(results)},
    )


def postprocess_transcription(result: TranscriptionResult, options: PostProcessOptions | None = None) -> TranscriptionResult:
    opts = options or PostProcessOptions()
    processed_segments: list[Segment] = []

    for segment in result.segments:
        text = _normalize_segment_text(segment.text, opts)
        if not text:
            continue

        if opts.dedupe_adjacent_segments and processed_segments:
            prev = processed_segments[-1]
            if _is_near_duplicate(prev.text, text, opts.adjacent_segment_similarity_threshold):
                prev.end = max(prev.end or 0.0, segment.end or prev.end or 0.0)
                continue

        processed_segments.append(
            Segment(id=len(processed_segments), start=segment.start, end=segment.end, text=text)
        )

    return TranscriptionResult(
        text=" ".join(seg.text for seg in processed_segments if seg.text).strip(),
        language=result.language,
        duration=result.duration,
        words=result.words,
        segments=processed_segments,
        raw_response=result.raw_response,
    )


def _normalize_segment_text(text: str, options: PostProcessOptions) -> str:
    normalized = text.strip()
    if not normalized:
        return ""
    if options.collapse_music_tokens:
        normalized = _collapse_music_tokens(normalized)
    if options.dedupe_repeated_phrases:
        normalized = _limit_consecutive_phrase_repeats(normalized, options.max_consecutive_phrase_repeats)
        normalized = _limit_repeated_japanese_phrases(normalized, options.max_consecutive_phrase_repeats)
    normalized = _limit_char_runs(normalized, options.max_consecutive_char_run)
    return normalized.strip()


def _collapse_music_tokens(text: str) -> str:
    collapsed = re.sub(r"(?:♪~?|♫|♬)(?:\s*(?:♪~?|♫|♬))+", "♪", text)
    collapsed = re.sub(r"\s+", " ", collapsed)
    return collapsed


def _limit_consecutive_phrase_repeats(text: str, max_repeat: int) -> str:
    if max_repeat < 1:
        return text
    tokens = text.split()
    if len(tokens) < 2:
        return text

    out: list[str] = []
    idx = 0
    while idx < len(tokens):
        matched = False
        for n in range(min(6, len(tokens) - idx), 1, -1):
            phrase = tokens[idx : idx + n]
            repeats = 1
            while idx + (repeats + 1) * n <= len(tokens) and tokens[idx + repeats * n : idx + (repeats + 1) * n] == phrase:
                repeats += 1
            if repeats > 1:
                out.extend(phrase * min(repeats, max_repeat))
                idx += repeats * n
                matched = True
                break
        if not matched:
            out.append(tokens[idx])
            idx += 1
    return " ".join(out)


def _limit_repeated_japanese_phrases(text: str, max_repeat: int) -> str:
    if max_repeat < 1:
        return text
    output = text
    for phrase_len in range(12, 1, -1):
        pattern = re.compile(rf"(.{{{phrase_len}}})(?:\1)+")
        while True:
            match = pattern.search(output)
            if not match:
                break
            phrase = match.group(1)
            repeats = len(match.group(0)) // phrase_len
            replacement = phrase * min(repeats, max_repeat)
            output = f"{output[:match.start()]}{replacement}{output[match.end():]}"
    return output


def _limit_char_runs(text: str, max_run: int) -> str:
    if max_run < 1:
        return text
    return re.sub(rf"(.)\1{{{max_run},}}", lambda m: m.group(1) * max_run, text)


def _is_near_duplicate(left: str, right: str, threshold: float) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    l_tokens = left.split()
    r_tokens = right.split()
    if not l_tokens or not r_tokens:
        return False
    common = sum(1 for i, tok in enumerate(l_tokens[: min(len(l_tokens), len(r_tokens))]) if tok == r_tokens[i])
    ratio = common / max(len(l_tokens), len(r_tokens))
    return ratio >= threshold


def split_pcm_by_size(audio_bytes: bytes, sample_rate_hz: int, max_chunk_bytes: int) -> tuple[list[bytes], list[float]]:
    if max_chunk_bytes <= 0 or len(audio_bytes) <= max_chunk_bytes:
        return [audio_bytes], [0.0]

    bytes_per_second = sample_rate_hz * 2
    aligned_max = max((max_chunk_bytes // 2) * 2, 2)

    chunks: list[bytes] = []
    offsets: list[float] = []
    start = 0
    while start < len(audio_bytes):
        end = min(start + aligned_max, len(audio_bytes))
        end = (end // 2) * 2
        if end <= start:
            end = min(start + 2, len(audio_bytes))
        chunks.append(audio_bytes[start:end])
        offsets.append(start / bytes_per_second)
        start = end

    return chunks, offsets


def _vad_segments(audio_bytes: bytes, sample_rate_hz: int, *, max_vad_segment_seconds: float) -> list[VADChunk]:
    samples = array("h")
    samples.frombytes(audio_bytes)
    if not samples:
        return [VADChunk(0, 0)]

    supported_rates = {8000, 16000, 32000, 48000}
    has_webrtcvad = importlib.util.find_spec("webrtcvad") is not None
    has_pkg_resources = importlib.util.find_spec("pkg_resources") is not None
    if sample_rate_hz not in supported_rates or not has_webrtcvad or not has_pkg_resources:
        return _vad_segments_energy(samples, sample_rate_hz, max_vad_segment_seconds=max_vad_segment_seconds)

    webrtcvad = importlib.import_module("webrtcvad")
    vad = webrtcvad.Vad(2)
    frame_ms = 30
    frame_size = int(sample_rate_hz * frame_ms / 1000)
    silence_limit_frames = max(1, int(300 / frame_ms))

    chunks: list[VADChunk] = []
    in_speech = False
    speech_start = 0
    silence_frames = 0

    for frame_start in range(0, len(samples), frame_size):
        frame_end = min(frame_start + frame_size, len(samples))
        frame = samples[frame_start:frame_end]
        if len(frame) < frame_size:
            frame.extend([0] * (frame_size - len(frame)))
        is_speech = vad.is_speech(frame.tobytes(), sample_rate_hz)

        if is_speech:
            if not in_speech:
                speech_start = frame_start
                in_speech = True
            silence_frames = 0
            continue

        if in_speech:
            silence_frames += 1
            if silence_frames >= silence_limit_frames:
                chunks.append(VADChunk(speech_start, frame_end))
                in_speech = False

    if in_speech:
        chunks.append(VADChunk(speech_start, len(samples)))

    if not chunks:
        chunks = [VADChunk(0, len(samples))]

    return _postprocess_vad_chunks(chunks, len(samples), sample_rate_hz, max_vad_segment_seconds=max_vad_segment_seconds)


def _vad_segments_energy(samples: array, sample_rate_hz: int, *, max_vad_segment_seconds: float) -> list[VADChunk]:
    frame_ms = 30
    frame_size = max(1, int(sample_rate_hz * frame_ms / 1000))
    energies: list[float] = []
    frame_ranges: list[tuple[int, int]] = []

    for frame_start in range(0, len(samples), frame_size):
        frame_end = min(frame_start + frame_size, len(samples))
        frame = samples[frame_start:frame_end]
        energy = sum(abs(v) for v in frame) / max(1, len(frame))
        energies.append(energy)
        frame_ranges.append((frame_start, frame_end))

    sorted_energy = sorted(energies)
    noise_floor = sorted_energy[max(0, int(len(sorted_energy) * 0.2) - 1)] if sorted_energy else 0.0
    threshold = max(250.0, noise_floor * 3.0)

    chunks: list[VADChunk] = []
    in_speech = False
    speech_start = 0
    silence_frames = 0
    for idx, energy in enumerate(energies):
        if energy >= threshold:
            if not in_speech:
                speech_start = frame_ranges[idx][0]
                in_speech = True
            silence_frames = 0
            continue

        if in_speech:
            silence_frames += 1
            if silence_frames >= 8:
                speech_end = frame_ranges[idx][1]
                chunks.append(VADChunk(speech_start, speech_end))
                in_speech = False

    if in_speech:
        chunks.append(VADChunk(speech_start, len(samples)))

    if not chunks:
        chunks = [VADChunk(0, len(samples))]

    return _postprocess_vad_chunks(chunks, len(samples), sample_rate_hz, max_vad_segment_seconds=max_vad_segment_seconds)


def _postprocess_vad_chunks(chunks: list[VADChunk], sample_count: int, sample_rate_hz: int, *, max_vad_segment_seconds: float) -> list[VADChunk]:
    padded: list[VADChunk] = []
    pad = int(0.15 * sample_rate_hz)
    for chunk in chunks:
        start = max(0, chunk.start_sample - pad)
        end = min(sample_count, chunk.end_sample + pad)
        if padded and start <= padded[-1].end_sample + int(0.2 * sample_rate_hz):
            padded[-1].end_sample = max(padded[-1].end_sample, end)
        else:
            padded.append(VADChunk(start, end))

    max_samples = max(int(max_vad_segment_seconds * sample_rate_hz), 1)
    split_chunks: list[VADChunk] = []
    for chunk in padded:
        start = chunk.start_sample
        while start < chunk.end_sample:
            end = min(start + max_samples, chunk.end_sample)
            split_chunks.append(VADChunk(start, end))
            start = end

    return split_chunks


def _build_packed_audio_batches(
    audio_bytes: bytes,
    sample_rate_hz: int,
    chunks: list[VADChunk],
    *,
    max_request_bytes: int,
) -> list[tuple[bytes, list[PackedPlacement]]]:
    bytes_per_sample = 2
    slot_seconds = 30
    slot_bytes = sample_rate_hz * bytes_per_sample * slot_seconds
    if max_request_bytes < slot_bytes:
        raise RivaTranscriptionError(
            f"ASR_MAX_CHUNK_MB too small for one 30-second packed slot ({slot_bytes} bytes required)"
        )
    slots_per_request = max(1, max_request_bytes // max(slot_bytes, 1))

    batches: list[tuple[bytes, list[PackedPlacement]]] = []
    for offset in range(0, len(chunks), slots_per_request):
        batch_chunks = chunks[offset : offset + slots_per_request]
        total_slots = max(1, len(batch_chunks))
        packed = bytearray(total_slots * slot_bytes)
        placements: list[PackedPlacement] = []

        for index, chunk in enumerate(batch_chunks):
            packed_start_sec = float(slot_seconds * index)
            packed_start_sample = int(packed_start_sec * sample_rate_hz)

            src_start = chunk.start_sample * bytes_per_sample
            src_end = chunk.end_sample * bytes_per_sample
            segment_bytes = audio_bytes[src_start:src_end]
            dst_start = packed_start_sample * bytes_per_sample
            slot_end = min(dst_start + slot_bytes, len(packed))
            dst_end = min(dst_start + len(segment_bytes), slot_end)
            copy_size = max(0, dst_end - dst_start)
            packed[dst_start:dst_end] = segment_bytes[:copy_size]

            duration = copy_size / (sample_rate_hz * bytes_per_sample)
            placements.append(
                PackedPlacement(
                    original_start=chunk.start_sample / sample_rate_hz,
                    packed_start=packed_start_sec,
                    duration=duration,
                    slot_duration=float(slot_seconds),
                )
            )

        packed_bytes = bytes(packed)
        if len(packed_bytes) > max_request_bytes:
            raise RivaTranscriptionError(
                f"Packed VAD batch size {len(packed_bytes)} exceeds max request bytes {max_request_bytes}"
            )
        batches.append((packed_bytes, placements))

    return batches


def _map_word(word: WordTimestamp, placements: list[PackedPlacement]) -> WordTimestamp | None:
    mapped_start = _map_time(word.start, placements)
    mapped_end = _map_time(word.end, placements)
    if mapped_start is None and mapped_end is None:
        return None
    return WordTimestamp(word=word.word, start=mapped_start, end=mapped_end, speaker=word.speaker)


def _map_segment(segment: Segment, placements: list[PackedPlacement]) -> Segment | None:
    mapped_start = _map_time(segment.start, placements)
    mapped_end = _map_time(segment.end, placements)
    if mapped_start is None and mapped_end is None:
        return None
    return Segment(id=segment.id, start=mapped_start, end=mapped_end, text=segment.text)


def _map_time(value: float | None, placements: list[PackedPlacement]) -> float | None:
    if value is None:
        return None
    for placement in placements:
        lower = placement.packed_start
        upper = placement.packed_start + placement.slot_duration
        if lower <= value <= upper + 1e-6:
            relative = min(max(0.0, value - placement.packed_start), placement.duration)
            return placement.original_start + relative
    return None


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
