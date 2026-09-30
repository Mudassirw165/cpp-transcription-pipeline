#include "whisper.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

struct Segment {
    double start;
    double end;
    std::string text;
};

static std::string shell_quote(const std::string &s) {
    std::string result = "'";
    for (char c : s) {
        if (c == '\'') result += "'\\''";
        else result += c;
    }
    return result + "'";
}

static std::string json_escape(const std::string &s) {
    std::ostringstream out;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) out << ' ';
                else out << c;
        }
    }
    return out.str();
}

static double probe_duration(const std::string &input) {
    const std::string command =
        "ffprobe -v error -show_entries format=duration "
        "-of default=noprint_wrappers=1:nokey=1 " + shell_quote(input);

    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Could not start ffprobe. Check that FFmpeg is installed.");
    }

    char buffer[256];
    std::string value;
    while (fgets(buffer, sizeof(buffer), pipe)) {
        value += buffer;
    }

    const int status = pclose(pipe);
    if (status != 0 || value.empty()) {
        throw std::runtime_error("ffprobe could not read the input audio.");
    }

    return std::stod(value);
}

// Reads the mono, 16 kHz, PCM16 WAV chunks created by FFmpeg.
static std::vector<float> read_pcm16_wav(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open decoded audio chunk: " + path.string());
    }

    char riff[4];
    char wave[4];
    uint32_t riff_size = 0;

    in.read(riff, 4);
    in.read(reinterpret_cast<char *>(&riff_size), 4);
    in.read(wave, 4);

    if (!in || std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("FFmpeg produced an invalid WAV file.");
    }

    uint16_t format = 0;
    uint16_t channels = 0;
    uint16_t bits = 0;
    uint32_t sample_rate = 0;
    uint32_t data_size = 0;

    while (in && data_size == 0) {
        char chunk_id[4];
        uint32_t chunk_size = 0;

        in.read(chunk_id, 4);
        in.read(reinterpret_cast<char *>(&chunk_size), 4);
        if (!in) break;

        if (std::string(chunk_id, 4) == "fmt ") {
            in.read(reinterpret_cast<char *>(&format), 2);
            in.read(reinterpret_cast<char *>(&channels), 2);
            in.read(reinterpret_cast<char *>(&sample_rate), 4);
            in.seekg(6, std::ios::cur); // byte rate and block alignment
            in.read(reinterpret_cast<char *>(&bits), 2);

            if (chunk_size > 16) {
                in.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (std::string(chunk_id, 4) == "data") {
            data_size = chunk_size;
            break;
        } else {
            in.seekg(chunk_size, std::ios::cur);
        }

        if (chunk_size & 1) {
            in.seekg(1, std::ios::cur);
        }
    }

    if (format != 1 || channels != 1 || sample_rate != 16000 ||
        bits != 16 || data_size == 0) {
        throw std::runtime_error("Expected mono 16 kHz PCM16 audio from FFmpeg.");
    }

    std::vector<int16_t> pcm(data_size / sizeof(int16_t));
    in.read(reinterpret_cast<char *>(pcm.data()), data_size);
    if (!in) {
        throw std::runtime_error("The decoded WAV chunk is incomplete.");
    }

    std::vector<float> samples;
    samples.reserve(pcm.size());
    for (int16_t sample : pcm) {
        samples.push_back(sample / 32768.0f);
    }
    return samples;
}

int main(int argc, char **argv) try {
    std::string input;
    std::string model;
    std::string output;
    std::string requested_language;
    int chunk_seconds = 600;
    bool mock = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto next_value = [&]() -> std::string {
            if (++i >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            return argv[i];
        };

        if (arg == "--model") {
            model = next_value();
        } else if (arg == "-o" || arg == "--output") {
            output = next_value();
        } else if (arg == "--language") {
            requested_language = next_value();
        } else if (arg == "--chunk-seconds") {
            chunk_seconds = std::stoi(next_value());
        } else if (arg == "--mock") {
            mock = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: transcribe_cpp AUDIO --model MODEL.bin "
                << "[--language en] [-o result.json] "
                << "[--chunk-seconds 600] [--mock]\n";
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("Unknown option: " + arg);
        } else if (input.empty()) {
            input = arg;
        } else {
            throw std::runtime_error("Only one input audio file is accepted.");
        }
    }

    if (input.empty() || (!mock && model.empty())) {
        throw std::runtime_error(
            "Provide an audio file and --model MODEL.bin. "
            "The model can be omitted with --mock."
        );
    }
    if (chunk_seconds < 1) {
        throw std::runtime_error("--chunk-seconds must be positive.");
    }
    if (!fs::is_regular_file(input)) {
        throw std::runtime_error("Input audio file not found: " + input);
    }

    const double duration = probe_duration(input);
    std::vector<Segment> segments;
    std::string language = requested_language.empty()
        ? "unknown"
        : requested_language;

    if (mock) {
        segments.push_back({
            0.0,
            std::min(2.0, duration),
            "Mock transcript for pipeline demonstration."
        });
        language = "en";
    } else {
        whisper_context_params context_params = whisper_context_default_params();
        whisper_context *ctx =
            whisper_init_from_file_with_params(model.c_str(), context_params);

        if (!ctx) {
            throw std::runtime_error("Could not load whisper.cpp model: " + model);
        }

        const fs::path temp_dir =
            fs::temp_directory_path() /
            ("transcribe-cpp-" + std::to_string(std::rand()));

        fs::create_directories(temp_dir);

        try {
            for (int index = 0;
                 index * static_cast<double>(chunk_seconds) < duration;
                 ++index) {
                const double offset = index * static_cast<double>(chunk_seconds);
                const double length =
                    std::min<double>(chunk_seconds, duration - offset);

                const fs::path wav =
                    temp_dir / ("chunk_" + std::to_string(index) + ".wav");

                const std::string command =
                    "ffmpeg -v error -y -ss " + std::to_string(offset) +
                    " -i " + shell_quote(input) +
                    " -t " + std::to_string(length) +
                    " -vn -ac 1 -ar 16000 -c:a pcm_s16le " +
                    shell_quote(wav.string());

                if (std::system(command.c_str()) != 0) {
                    throw std::runtime_error(
                        "FFmpeg failed while decoding chunk " +
                        std::to_string(index)
                    );
                }

                const std::vector<float> samples = read_pcm16_wav(wav);

                whisper_full_params params =
                    whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

                params.n_threads = static_cast<int>(
                    std::max(1u, std::thread::hardware_concurrency())
                );

                if (requested_language.empty()) {
                    params.language = "auto";
                    params.detect_language = true;
                } else {
                    params.language = requested_language.c_str();
                    params.detect_language = false;
                }

                params.print_progress = false;
                params.print_realtime = false;
                params.print_timestamps = false;

                const int result = whisper_full(
                    ctx,
                    params,
                    samples.data(),
                    static_cast<int>(samples.size())
                );

                if (result != 0) {
                    throw std::runtime_error(
                        "whisper.cpp inference failed on chunk " +
                        std::to_string(index)
                    );
                }

                if (language == "unknown") {
                    const char *detected =
                        whisper_lang_str(whisper_full_lang_id(ctx));
                    if (detected) {
                        language = detected;
                    }
                }

                const int segment_count = whisper_full_n_segments(ctx);
                for (int s = 0; s < segment_count; ++s) {
                    const double start =
                        offset + whisper_full_get_segment_t0(ctx, s) / 100.0;
                    const double end =
                        offset + whisper_full_get_segment_t1(ctx, s) / 100.0;

                    segments.push_back({
                        std::min(start, duration),
                        std::min(end, duration),
                        whisper_full_get_segment_text(ctx, s)
                    });
                }

                fs::remove(wav);
            }
        } catch (...) {
            fs::remove_all(temp_dir);
            whisper_free(ctx);
            throw;
        }

        fs::remove_all(temp_dir);
        whisper_free(ctx);

        if (segments.empty()) {
            throw std::runtime_error(
                "No speech segments were recognized. Check that the audio is "
                "audible and retry with --language en (or the correct language code)."
            );
        }
    }

    std::ostringstream json;
    json << std::fixed << std::setprecision(3)
         << "{\n  \"source\": \""
         << json_escape(fs::path(input).filename().string())
         << "\",\n  \"language\": \"" << json_escape(language)
         << "\",\n  \"duration_seconds\": " << duration
         << ",\n  \"segments\": [";

    std::string all_text;
    for (size_t i = 0; i < segments.size(); ++i) {
        const Segment &segment = segments[i];

        json << (i ? "," : "")
             << "\n    {\"start\": " << segment.start
             << ", \"end\": " << segment.end
             << ", \"text\": \"" << json_escape(segment.text) << "\"}";

        if (!all_text.empty()) {
            all_text += " ";
        }
        all_text += segment.text;
    }

    json << (segments.empty() ? "" : "\n  ")
         << "],\n  \"text\": \"" << json_escape(all_text)
         << "\",\n  \"backend\": \"" << (mock ? "mock" : "whisper.cpp")
         << "\"\n}\n";

    if (output.empty()) {
        std::cout << json.str();
    } else {
        std::ofstream out(output);
        if (!out) {
            throw std::runtime_error("Cannot write output file: " + output);
        }
        out << json.str();
    }

    return 0;
} catch (const std::exception &e) {
    std::cerr << "transcription failed: " << e.what() << '\n';
    return 1;
}