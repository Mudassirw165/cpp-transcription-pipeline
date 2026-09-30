#!/usr/bin/env python3
"""Transcribe an audio file to timestamped JSON using faster-whisper."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg is required for audio decoding. Install it and ensure it is on PATH.")
    return path


def audio_duration(ffmpeg: str, path: Path) -> float:
    # Decode to mono 16 kHz PCM first; the model accepts this normalized format.
    # ffmpeg's -i parsing is sufficient here and avoids an extra probing dependency.
    result = subprocess.run(
        [ffmpeg, "-i", str(path), "-f", "null", "-"], capture_output=True, text=True
    )
    import re

    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError(f"Could not read audio duration or decode input: {path}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def transcribe(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.audio).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {source}")
    ffmpeg = require_ffmpeg()
    duration = audio_duration(ffmpeg, source)

    # Convert any ffmpeg-supported input (WAV, MP3, M4A, FLAC, OGG, ...) to a
    # uniform streamable intermediate, then process bounded windows so long
    # recordings do not need to be held as one decoded array in memory.
    with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
        normalized = Path(tmp) / "audio.wav"
        subprocess.run(
            [ffmpeg, "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(normalized)],
            check=True, capture_output=True,
        )

        if args.mock:
            return {
                "source": source.name,
                "language": args.language or "en",
                "duration_seconds": round(duration, 3),
                "segments": [{"start": 0.0, "end": round(min(duration, 2.0), 3),
                              "text": "Mock transcript for pipeline demonstration."}],
                "text": "Mock transcript for pipeline demonstration.",
                "backend": "mock",
            }

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        # ffmpeg segment muxer creates bounded WAV chunks without loading the
        # entire recording into Python. Each window resets model context.
        chunk_dir = Path(tmp) / "chunks"
        chunk_dir.mkdir()
        subprocess.run(
            [ffmpeg, "-y", "-i", str(normalized), "-f", "segment", "-segment_time",
             str(args.chunk_seconds), "-reset_timestamps", "1", "-acodec", "pcm_s16le",
             str(chunk_dir / "chunk_%05d.wav")],
            check=True, capture_output=True,
        )

        segments: list[dict[str, Any]] = []
        detected_language = args.language
        for index, chunk in enumerate(sorted(chunk_dir.glob("chunk_*.wav"))):
            offset = index * args.chunk_seconds
            chunk_segments, info = model.transcribe(
                str(chunk), language=args.language, beam_size=args.beam_size,
                vad_filter=args.vad_filter,
            )
            if detected_language is None:
                detected_language = info.language
            for seg in chunk_segments:
                start = min(duration, offset + float(seg.start))
                end = min(duration, offset + float(seg.end))
                segments.append({"start": round(start, 3), "end": round(end, 3),
                                 "text": seg.text.strip()})

    return {
        "source": source.name,
        "language": detected_language,
        "duration_seconds": round(duration, 3),
        "segments": segments,
        "text": " ".join(s["text"] for s in segments).strip(),
        "backend": "faster-whisper",
        "model": args.model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Input audio file")
    parser.add_argument("-o", "--output", help="JSON output path (defaults to stdout)")
    parser.add_argument("--model", default="small", help="faster-whisper model name (default: small)")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--compute-type", default="int8", help="e.g. int8, float16, float32")
    parser.add_argument("--language", help="Language code; omit for automatic detection")
    parser.add_argument("--chunk-seconds", type=int, default=600,
                        help="Maximum chunk duration for long recordings (default: 600)")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad-filter", action="store_true", help="Filter non-speech intervals")
    parser.add_argument("--mock", action="store_true", help="Return a demonstration result without model inference")
    args = parser.parse_args()
    if args.chunk_seconds < 1:
        parser.error("--chunk-seconds must be positive")
    try:
        result = transcribe(args)
        payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)
        return 0
    except Exception as exc:
        print(f"transcription failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
