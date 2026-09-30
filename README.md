# C++ Audio Transcription Pipeline

A command-line program that transcribes audio with [whisper.cpp](https://github.com/ggml-org/whisper.cpp). It writes a JSON file containing the transcript and timestamped speech segments.

## What it does

- Accepts audio file paths, including WAV, MP3, and OGG.
- Uses FFmpeg to convert audio to mono, 16 kHz PCM before transcription.
- Splits long recordings into chunks. The default chunk size is 10 minutes.
- Returns the language, full transcript, and each segment’s start and end time in seconds.
- Includes sample recordings in the `samples/` folder.

The model’s context resets at chunk boundaries.

## Requirements

- macOS with Apple Command Line Tools
- CMake and FFmpeg
- whisper.cpp
- A Whisper model file

Install the tools with Homebrew:

```sh
xcode-select --install
brew install cmake ffmpeg git
```

If Apple Command Line Tools are already installed, continue to the next step.

## Build whisper.cpp and download a model

These commands build and install whisper.cpp and download the multilingual `base` model:

```sh
mkdir -p ~/Developer
cd ~/Developer
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
sh ./models/download-ggml-model.sh base
cmake -B build -DCMAKE_INSTALL_PREFIX="$HOME/Developer/whisper-install"
cmake --build build -j 4 --config Release
cmake --install build
```

The multilingual model can transcribe multiple languages. To use an English-only model instead, download `base.en` and use `ggml-base.en.bin` in the commands below.

## Build this program

Open a terminal in the folder containing this README and `CMakeLists.txt`, then run:

```sh
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$HOME/Developer/whisper-install"
cmake --build build --config Release
```

## Run a transcription

For the included English sample `DIALOGUE.ogg`, run:

```sh
./build/transcribe_cpp samples/DIALOGUE.ogg \
  --model "$HOME/Developer/whisper.cpp/models/ggml-base.en.bin" \
  --language en \
  --output transcript.json
```

For another language, use the multilingual model and provide its language code. For example, Spanish uses `es`:

```sh
./build/transcribe_cpp samples/mjn.wav \
  --model "$HOME/Developer/whisper.cpp/models/ggml-base.bin" \
  --language es \
  --output transcript.json
```

Replace `es` with the correct language code, or leave out `--language` to let the model try to detect it.

The program saves `transcript.json` in the current folder. The JSON includes `text` for the combined transcript and a `segments` array with the text and start/end timestamps for each segment.

## Example output

```json
{
  "source": "DIALOGUE.ogg",
  "language": "en",
  "duration_seconds": 12.5,
  "segments": [
    {
      "start": 0.0,
      "end": 2.31,
      "text": "Example spoken words."
    }
  ],
  "text": "Example spoken words.",
  "backend": "whisper.cpp"
}
```

## Try your own audio

You can pass an audio file path instead of a sample path:

```sh
./build/transcribe_cpp "$HOME/Downloads/your-audio.wav" \
  --model "$HOME/Developer/whisper.cpp/models/ggml-base.bin" \
  --output transcript.json
```

Use `--chunk-seconds 300` to process five-minute chunks instead of the default ten-minute chunks.
