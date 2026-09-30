# Timestamped transcription pipeline

A small Python CLI that accepts an audio file, normalizes it with FFmpeg, transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and emits JSON with timestamped segments and combined text.

A native C++ implementation is also included. It uses [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and the same JSON contract.

## C++ build and run

Install FFmpeg (including `ffmpeg` and `ffprobe`) and build/install whisper.cpp so CMake can find its `whisper` package. Then:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/whisper.cpp/install
cmake --build build --config Release
./build/transcribe_cpp recording.mp3 --model /path/to/ggml-small.bin --language en --output transcript.json
```

The model file is downloaded separately from the whisper.cpp model collection. Use `--chunk-seconds 300` to reduce per-chunk memory use. `--mock` skips model loading and inference while still validating/decoding input metadata:

```sh
./build/transcribe_cpp recording.wav --mock
```

In mock mode the program probes the audio metadata and emits a sample payload; real decoding happens when inference is enabled.

## Run it

Requirements: Python 3.9+, FFmpeg on `PATH`, and `pip install -r requirements.txt`. The first run may download the selected model.

```sh
python transcribe.py recording.mp3 --output transcript.json
python transcribe.py interview.wav --language en --model small --vad-filter
```

Use `--device cuda --compute-type float16` where a compatible GPU is available. `--mock` outputs a sample response for integration wiring without downloading or running a model; it still validates and normalizes the audio input.

## Output

```json
{
  "source": "recording.mp3",
  "language": "en",
  "duration_seconds": 12.5,
  "segments": [
    {"start": 0.0, "end": 2.31, "text": "Example spoken words."}
  ],
  "text": "Example spoken words.",
  "backend": "faster-whisper",
  "model": "small"
}
```

## Engineering choices

- **Formats:** FFmpeg decodes common containers and codecs (including WAV, MP3, M4A, FLAC, and OGG) into mono 16 kHz PCM. Decode failures are returned as a nonzero CLI exit with a useful error.
- **Long recordings:** Normalized audio is segmented into bounded windows (`--chunk-seconds`, 10 minutes by default), so inference only loads one chunk at a time. Segment timestamps are shifted by each chunk's offset. Context resets at chunk boundaries; choose a longer window when continuity matters and ensure the chunk duration is not so large that it creates memory pressure.
- **Timestamps:** `faster-whisper` supplies segment start/end times relative to each window. The pipeline converts them to source-relative seconds and caps the final endpoint at the probed recording duration.
- **Downstream contract:** JSON contains both structured segments and a joined `text` field; timestamps are numeric seconds and text is UTF-8.

For production use, wrap the function with an upload API/job queue, impose file size and duration limits, clean up temporary files on cancellation, and persist model/version metadata alongside outputs.

The C++ implementation uses `ffprobe` to read duration and invokes `ffmpeg` once per bounded window. It keeps only one decoded window in memory. Its model timestamps are provided in centiseconds by whisper.cpp and converted to source-relative seconds. Chunk boundaries reset decoder context, as in the Python implementation.
