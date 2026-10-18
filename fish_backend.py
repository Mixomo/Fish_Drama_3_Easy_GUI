import gradio as gr
import gc
import json
import os
import random
import re
import shutil
import soundfile as sf
import threading
import time
import warnings
import wave
from pathlib import Path

import numpy as np

try:
    from starlette.exceptions import StarletteDeprecationWarning
    warnings.filterwarnings(
        "ignore",
        message=("'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated. "
                 "Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead."),
        category=StarletteDeprecationWarning,
    )
except ImportError:
    pass

try:
    import winsound
except ImportError:
    winsound = None

from fish_easy import drama3_runtime
from fish_easy.console import log

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
COMPILE_CACHE_DIR = os.path.join(MODELS_DIR, ".cache")
WHISPER_MODELS_DIR = os.path.join(MODELS_DIR, "whisper")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
SAMPLES_DIR = os.path.join(ROOT_DIR, "samples")
SAMPLE_PREVIEW_DIR = os.path.join(COMPILE_CACHE_DIR, "sample_previews")
for _directory in (MODELS_DIR, COMPILE_CACHE_DIR, WHISPER_MODELS_DIR,
                   OUTPUTS_DIR, SAMPLES_DIR, SAMPLE_PREVIEW_DIR):
    os.makedirs(_directory, exist_ok=True)

_inference_cancel_event = threading.Event()
_aux_cancel_event = threading.Event()


def begin_aux_job():
    _aux_cancel_event.clear()


def stop_aux_job():
    _aux_cancel_event.set()
    return "Stop requested. The active transcription operation will stop at the next safe boundary."


def _aux_cancelled():
    return _aux_cancel_event.is_set()


def begin_inference():
    _inference_cancel_event.clear()


def inference_cancelled():
    return _inference_cancel_event.is_set()


CHIME_PATH = os.path.join(ROOT_DIR, "assets", "inference_training_done.wav")


def play_done_chime():
    if os.path.exists(CHIME_PATH) and winsound is not None:
        try:
            winsound.PlaySound(CHIME_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            print(f"Could not play completion chime: {exc}")
    elif winsound is not None:
        winsound.MessageBeep()


def normalize_seed(seed_value):
    if seed_value is None or seed_value == "":
        return None
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        return None
    if seed == 0:
        return None
    return min(abs(seed), (1 << 32) - 1)


def resolve_seed_for_generation(seed_value):
    normalized_seed = normalize_seed(seed_value)
    return normalized_seed if normalized_seed is not None else random_seed_value()


def random_seed_value():
    return random.randint(1, (1 << 32) - 1)


def _console_progress(label, current, total, detail="", width=26):
    try:
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        pct = current / total
        filled = int(round(width * pct))
        bar = "█" * filled + "░" * (width - filled)
        suffix = f" · {detail}" if detail else ""
        print(f"[UI-PROGRESS] {label:<18} [{bar}] {pct*100:6.2f}% ({current}/{total}){suffix}", end="\r", flush=True)
        if current >= total:
            print()
    except Exception:
        pass


def _phase(progress, fraction, message):
    fraction = max(0.0, min(1.0, float(fraction)))
    try:
        progress(fraction, desc=str(message))
    except Exception:
        pass
    _console_progress("Workflow", int(round(fraction * 100)), 100, str(message), width=24)

# --- Drama 3 / transcription settings ---
GGUF_MODELS = drama3_runtime.CPP_VARIANTS
CPP_BACKENDS = drama3_runtime.available_cpp_backends()
SPEECH_BACKENDS = drama3_runtime.available_speech_backends()
WHISPER_LANGS = {"Auto-detect": None}
WHISPER_MODELS = {
    "large-v3 (~10 GB VRAM)": "large-v3",
    "large-v2 (~10 GB VRAM)": "large-v2",
    "medium (~5 GB VRAM)": "medium",
    "small (~2 GB VRAM)": "small",
    "base (~1 GB VRAM)": "base",
    "tiny (~1 GB VRAM)": "tiny",
}
NONE = "None"
CHUNK_CHOICES = ["None", "Paragraph/Sentence Auto", "Periods", "Paragraphs", "Lines", "Speaker turns"]
def split_long_text(text, mode="None"):
    import re
    text = str(text or "").strip()
    if not text:
        return []
    mode = mode if mode in CHUNK_CHOICES else "None"
    if mode == "None":
        return [text]
    if mode == "Paragraphs":
        return [x.strip() for x in re.split(r"\n\s*\n+", text) if x.strip()] or [text]
    if mode == "Lines":
        return [x.strip() for x in text.splitlines() if x.strip()] or [text]
    if mode == "Periods":
        normalized = re.sub(r"\s+", " ", text)
        return [x.strip() for x in re.split(r"(?<=\.)\s+", normalized) if x.strip()] or [text]
    if mode == "Speaker turns":
        chunks, current = [], []
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if (re.match(r"^\s*\[?SPEAKER\d+\]?", clean, flags=re.I) or re.match(r"^<\|speaker:\d+\|>", clean)) and current:
                chunks.append("\n".join(current).strip()); current=[clean]
            else:
                current.append(clean)
        if current: chunks.append("\n".join(current).strip())
        return chunks or [text]
    chunks=[]
    paragraphs=[x.strip() for x in re.split(r"\n\s*\n+", text) if x.strip()] or [text]
    for paragraph in paragraphs:
        normalized=re.sub(r"[ \t]+", " ", paragraph).strip()
        sentences=[x.strip() for x in re.split(r"(?<=[.!?…。！？;；])\s*", normalized) if x.strip()]
        chunks.extend(sentences or [normalized])
    return chunks or [text]


def split_text_by_size(text, size, mode):
    """Apply the configurable semantic chunking rules used by both Drama 3 runtimes."""
    limit = max(1, int(size))
    base = split_long_text(text, mode)
    result = []
    if str(mode or "None") == "None":
        base = [str(text or "").strip()]
    for part in base:
        part = str(part).strip()
        while len(part) > limit:
            boundary = max(part.rfind(" ", 0, limit + 1),
                           part.rfind("\n", 0, limit + 1))
            if boundary < int(limit * 0.55):
                boundary = limit
            cut = part[:boundary].rstrip()
            if cut and cut[-1] not in ".!?…。！？;；,:，、—-" and boundary < len(part):
                cut += ","
            result.append(cut)
            part = part[boundary:].strip()
        if part:
            result.append(part)
    return result or ([str(text).strip()] if str(text or "").strip() else [])


def get_sample_choices():
    if not os.path.exists(SAMPLES_DIR): return []
    files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav")]
    return sorted([f.replace(".wav", "") for f in files])

def get_sample_preview_path(audio_path):
    """Create a browser-friendly dual-mono preview without modifying the source."""
    if not audio_path or not os.path.isfile(audio_path):
        return audio_path

    source_stat = os.stat(audio_path)
    source_key = f"{source_stat.st_mtime_ns}_{source_stat.st_size}"
    sample_stem = os.path.splitext(os.path.basename(audio_path))[0]
    preview_path = os.path.join(SAMPLE_PREVIEW_DIR, f"{sample_stem}_{source_key}.wav")
    if os.path.isfile(preview_path):
        return preview_path

    try:
        audio, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
        mono = audio[:, 0] if audio.shape[1] == 1 else np.mean(audio, axis=1)
        pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
        sf.write(
            preview_path,
            np.column_stack((pcm16, pcm16)),
            sample_rate,
            subtype="PCM_16",
        )

        prefix = f"{sample_stem}_"
        for cached_name in os.listdir(SAMPLE_PREVIEW_DIR):
            cached_path = os.path.join(SAMPLE_PREVIEW_DIR, cached_name)
            if cached_name.startswith(prefix) and cached_path != preview_path:
                try:
                    os.remove(cached_path)
                except OSError:
                    pass
        return preview_path
    except Exception as exc:
        print(f"[Sample Preview] Could not create dual-mono preview for {audio_path}: {exc}")
        return audio_path


def _trim_process_ram():
    """Best-effort return of unused Python/CRT pages to Windows."""
    if os.name != "nt":
        return
    try:
        import ctypes
        try:
            ctypes.CDLL("msvcrt")._heapmin()
        except Exception:
            pass
        try:
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            handle = kernel32.GetCurrentProcess()
            psapi.EmptyWorkingSet(handle)
        except Exception:
            pass
    except Exception:
        pass


def load_sample(sample_name):
    """Load sample audio and text. Returns (audio_path_or_None, text_str)."""
    if not sample_name:
        return None, ""
    audio_path = os.path.join(SAMPLES_DIR, f"{sample_name}.wav")
    txt_path = os.path.join(SAMPLES_DIR, f"{sample_name}.txt")
    json_path = os.path.join(SAMPLES_DIR, f"{sample_name}.json")
    
    text = ""
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = data.get("Text", "") or data.get("text", "")
        except: pass
    
    if not text and os.path.exists(txt_path):
        for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                with open(txt_path, "r", encoding=enc) as f:
                    text = f.read().strip()
                if text: break
            except: pass
            
    if os.path.exists(audio_path):
        return get_sample_preview_path(audio_path), text
    return None, text

def process_audio_array(audio_data):
    """Converts audio to mono and normalizes volume to [-1, 1]"""
    # Convert to mono if stereo
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize volume
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
        
    return audio_data


def write_synthesized_audio(path, audio_data, sample_rate):
    """Write PCM16 dual-mono so browser players reproduce the signal centered."""
    mono = process_audio_array(np.asarray(audio_data))
    pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
    dual_mono = np.column_stack((pcm16, pcm16))
    sf.write(path, dual_mono, sample_rate, subtype="PCM_16")


def begin_inference():
    """Reset the cooperative cancellation token for a new inference job."""
    _inference_cancel_event.clear()


def inference_cancelled():
    return _inference_cancel_event.is_set()


def generate_dialogue(
    cpp_model_str,
    cpp_backend,
    cpp_device,
    top_p,
    top_k,
    temp,
    seed,
    row_count,
    silence_duration,
    *args,
    max_tokens=500,
    context_length=4096,
    text_chunk_size=200,
    text_chunk_mode="tag_aware",
    cpu_expert_layers=24,
    codec_backend="same",
    chunk_mode="None",
    chunk_gap=0.0,
    multi_reference_cond=None,
    previous_takes=None,
    previous_takes_file=None,
    progress=gr.Progress(),
):
    # args is [sample1, ..., sample20, text1, ..., text20]
    num_max = 20 # Should match MAX_DIALOGUE_SEGMENTS
    samples = args[:num_max]
    texts = args[num_max:]
    
    segments = []
    for i in range(int(row_count)):
        s = samples[i]
        t = texts[i]
        if t:
            segments.append((s, t))
            
    if not segments:
        return None, "Please add at least one speaker and text.", normalize_seed(seed)
        
    output_path = os.path.join(OUTPUTS_DIR, f"dialogue_{int(time.time()*1000)}.wav")
    final_sr = None
    output_frames = 0
    silence_frames = 0
    wave_writer = None

    for i, (sample_name, text) in enumerate(segments):
        progress(i / len(segments), desc=f"Processing turn {i+1}/{len(segments)} ({sample_name or 'model voice'})...")
        _console_progress("Dialogue", i, len(segments), f"turn {i+1}/{len(segments)} · {sample_name or 'model voice'}")

        if sample_name and sample_name != NONE:
            ref_audio, ref_text = load_sample(sample_name)
            if not ref_audio:
                drama3_runtime.shutdown(clear_voice_cache=True)
                if wave_writer is not None:
                    wave_writer.close()
                if os.path.isfile(output_path):
                    os.remove(output_path)
                return None, f"Voice '{sample_name}' was not found for turn {i+1}.", normalize_seed(seed)
        else:
            ref_audio, ref_text = None, ""

        wav_path, status, _ = clone_voice(
            cpp_model_str, cpp_backend, cpp_device, text, ref_audio, ref_text,
            top_p, top_k, temp, seed, max_tokens=max_tokens,
            context_length=context_length, text_chunk_size=text_chunk_size,
            text_chunk_mode=text_chunk_mode, cpu_expert_layers=cpu_expert_layers,
            codec_backend=codec_backend, chunk_mode=chunk_mode, chunk_gap=chunk_gap,
            multi_reference_cond=multi_reference_cond, previous_takes=previous_takes,
            previous_takes_file=previous_takes_file, progress=progress,
        )
        if not wav_path or not os.path.isfile(wav_path):
            drama3_runtime.shutdown(clear_voice_cache=True)
            gc.collect()
            _trim_process_ram()
            if wave_writer is not None:
                wave_writer.close()
            if os.path.isfile(output_path):
                os.remove(output_path)
            return None, f"Turn {i+1} failed ({sample_name or 'model voice'}): {status}", normalize_seed(seed)

        try:
            audio_data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
            mono = np.mean(audio_data, axis=1, dtype=np.float32)
            del audio_data
            if final_sr is None:
                final_sr = int(sr)
                silence_frames = max(0, int(float(silence_duration or 0) * final_sr))
                wave_writer = wave.open(output_path, "wb")
                wave_writer.setnchannels(2)
                wave_writer.setsampwidth(2)
                wave_writer.setframerate(final_sr)
            elif int(sr) != final_sr:
                if wave_writer is not None:
                    wave_writer.close()
                if os.path.isfile(output_path):
                    os.remove(output_path)
                return None, f"Turn {i+1} sample rate changed from {final_sr} to {sr} Hz.", normalize_seed(seed)

            peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            if peak > 0:
                mono /= peak
            turn_frames = mono.size
            for offset in range(0, turn_frames, 65536):
                pcm = np.clip(mono[offset:offset + 65536] * 32767.0,
                              -32768, 32767).astype("<i2", copy=False)
                stereo = np.empty((pcm.size, 2), dtype="<i2")
                stereo[:, 0] = pcm
                stereo[:, 1] = pcm
                wave_writer.writeframesraw(stereo.tobytes())
            if i + 1 < len(segments) and silence_frames:
                silence_block = b"\x00\x00\x00\x00" * min(silence_frames, 8192)
                remaining = silence_frames
                while remaining:
                    block_frames = min(remaining, 8192)
                    wave_writer.writeframesraw(silence_block[:block_frames * 4])
                    remaining -= block_frames
            output_frames += turn_frames + (silence_frames if i + 1 < len(segments) else 0)
        except Exception as exc:
            if wave_writer is not None:
                wave_writer.close()
                wave_writer = None
            if os.path.isfile(output_path):
                os.remove(output_path)
            return None, f"Could not assemble turn {i+1}: {exc}", normalize_seed(seed)
        finally:
            if os.path.isfile(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
            if "audio_data" in locals():
                del audio_data
            if "mono" in locals():
                del mono
            if "pcm" in locals():
                del pcm
            if "stereo" in locals():
                del stereo
            gc.collect()
            drama3_runtime.shutdown(clear_voice_cache=True)
            _trim_process_ram()

        if _inference_cancel_event.is_set():
            if wave_writer is not None:
                wave_writer.close()
            if os.path.isfile(output_path):
                os.remove(output_path)
            return None, "Dialogue generation stopped after unloading the current turn.", normalize_seed(seed)

    if wave_writer is not None:
        wave_writer.close()
    if final_sr is not None and output_frames:
        _console_progress("Dialogue", len(segments), len(segments), "complete")
        return output_path, f"Dialogue generated successfully with {len(segments)} independent turns.", normalize_seed(seed)
    return None, "No audio generated.", normalize_seed(seed)

def transcribe_only(audio_path, model_size, language_name, batch_size=4, progress=gr.Progress()):
    begin_aux_job()
    if not audio_path:
        return ""
    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
    except ImportError:
        return "Error: faster-whisper module not found. Please run install.ps1 again."
        
    import ctranslate2
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    lang_code = WHISPER_LANGS.get(language_name)
    
    try:
        _phase(progress, 0.2, f"Loading Faster-Whisper {model_size} · {device}")
        
        whisper_cache = os.path.join(MODELS_DIR, "whisper")
        os.makedirs(whisper_cache, exist_ok=True)
        
        compute_type = "float16" if device == "cuda" else "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=whisper_cache)
        batched_model = BatchedInferencePipeline(model=model)
        
        _phase(progress, 0.5, f"Transcribing reference audio · batch {max(1, int(batch_size or 1))}")
        segments, info = batched_model.transcribe(
            audio_path, language=lang_code, batch_size=max(1, int(batch_size or 1)), beam_size=5
        )
        
        # Gather all text segments
        pieces = []
        for segment in segments:
            if _aux_cancelled():
                return "Transcription stopped by user."
            pieces.append(segment.text)
        text = " ".join(pieces).strip()
        
        # Unload from VRAM after transcription is done
        import gc
        del batched_model
        del model
        gc.collect()
            
        _phase(progress, 1.0, "Transcription complete")
        play_done_chime()
        return text
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Faster-Whisper Error: {str(e)}"

def save_prep_sample(audio_path, sample_name, transcription, language="auto"):
    if not audio_path:
        return "Please provide an audio file to save.", gr.update()
    if not sample_name or sample_name.strip() == "":
        sample_name = f"sample_{int(time.time())}"
        
    sample_name = "".join([c for c in sample_name if c.isalnum() or c in (" ", "_")]).replace(" ", "_").strip("_")
    
    dest_wav = os.path.join(SAMPLES_DIR, f"{sample_name}.wav")
    dest_json = os.path.join(SAMPLES_DIR, f"{sample_name}.json")
    
    try:
        shutil.copy2(audio_path, dest_wav)
        
        # Save JSON metadata pair as requested
        metadata = {
            "Type": "Sample",
            "Text": transcription.strip(),
            "Language": str(language or "auto")
        }
        with open(dest_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        # Cleanup gc and update choices
        import gc
        gc.collect()
        return f"Successfully saved sample: {sample_name} in {SAMPLES_DIR}", gr.update(choices=get_sample_choices(), value=sample_name)
    except Exception as e:
        return f"Error saving sample: {str(e)}", gr.update()

def delete_sample(sample_name):
    if not sample_name:
        return "No sample selected.", gr.update()
    try:
        dest_audio = os.path.join(SAMPLES_DIR, f"{sample_name}.wav")
        dest_txt = os.path.join(SAMPLES_DIR, f"{sample_name}.txt")
        if os.path.exists(dest_audio): os.remove(dest_audio)
        if os.path.exists(dest_txt): os.remove(dest_txt)
        dest_json = os.path.join(SAMPLES_DIR, f"{sample_name}.json")
        if os.path.exists(dest_json): os.remove(dest_json)
        return f"Deleted sample '{sample_name}'.", gr.update(choices=get_sample_choices(), value=None)
    except Exception as e:
        return f"Error deleting: {e}", gr.update()

def _clone_voice_drama3_chunked(
    cpp_model_str, cpp_backend, cpp_device, text, ref_audio, ref_text,
    top_p, top_k, temp, seed, max_tokens, context_length, text_chunk_size,
    text_chunk_mode, cpu_expert_layers, codec_backend, multi_reference_cond,
    previous_takes, previous_takes_file, chunk_mode, chunk_gap, progress,
):
    chunks = split_text_by_size(text, text_chunk_size, chunk_mode)
    if len(chunks) <= 1:
        return _clone_voice_drama3_single(
            cpp_model_str, cpp_backend, cpp_device, text, ref_audio, ref_text,
            top_p, top_k, temp, seed, max_tokens, context_length,
            text_chunk_size, text_chunk_mode, cpu_expert_layers, codec_backend,
            multi_reference_cond, previous_takes, previous_takes_file, progress,
        )
    outputs, sample_rate = [], None
    for index, chunk in enumerate(chunks):
        if _inference_cancel_event.is_set():
            return None, "Inference stopped between text chunks.", normalize_seed(seed)
        result = _clone_voice_drama3_single(
            cpp_model_str, cpp_backend, cpp_device, chunk, ref_audio, ref_text,
            top_p, top_k, temp, seed, max_tokens, context_length,
            text_chunk_size, text_chunk_mode, cpu_expert_layers, codec_backend,
            multi_reference_cond, previous_takes, previous_takes_file, progress,
        )
        path, status, _ = result
        if not path or not os.path.isfile(path):
            return None, f"Chunk {index + 1}/{len(chunks)} failed: {status}", normalize_seed(seed)
        audio_data, sample_rate = sf.read(path)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        outputs.append(np.asarray(audio_data, dtype=np.float32))
        if index + 1 < len(chunks) and float(chunk_gap or 0) > 0:
            outputs.append(np.zeros(int(sample_rate * float(chunk_gap)), dtype=np.float32))
        progress((index + 1) / len(chunks), desc=f"Drama 3 chunk {index + 1}/{len(chunks)}")
    final = os.path.join(OUTPUTS_DIR, f"drama3_chunks_{int(time.time()*1000)}.wav")
    write_synthesized_audio(final, np.concatenate(outputs), sample_rate)
    return final, f"Drama 3 synthesis completed ({len(chunks)} chunks).", normalize_seed(seed)


def clone_voice(
    cpp_model_str, cpp_backend, cpp_device, text, ref_audio, ref_text,
    top_p, top_k, temp, seed, max_tokens=500, context_length=4096,
    text_chunk_size=200, text_chunk_mode="tag_aware", cpu_expert_layers=24,
    codec_backend="same", chunk_mode="None", chunk_gap=0.0,
    multi_reference_cond=None, previous_takes=None, previous_takes_file=None,
    progress=gr.Progress(),
):
    if not str(text or "").strip():
        return None, "Please enter text to synthesize.", normalize_seed(seed)
    if ref_audio and not str(ref_text or "").strip():
        return None, "Reference transcript is required when reference audio is provided.", normalize_seed(seed)
    return _clone_voice_drama3_chunked(
        cpp_model_str, cpp_backend, cpp_device, str(text), ref_audio, ref_text,
        top_p, top_k, temp, resolve_seed_for_generation(seed), max_tokens,
        context_length, text_chunk_size, text_chunk_mode, cpu_expert_layers,
        codec_backend, multi_reference_cond, previous_takes, previous_takes_file,
        chunk_mode, chunk_gap, progress,
    )


def _clone_voice_drama3_single(
    cpp_model_str, cpp_backend, cpp_device, text, ref_audio, ref_text,
    top_p, top_k, temp, seed, max_tokens, context_length, text_chunk_size,
    text_chunk_mode, cpu_expert_layers, codec_backend, multi_reference_cond,
    previous_takes, previous_takes_file, progress,
):
    selected_seed = normalize_seed(seed)
    try:
        output_path, _ = drama3_runtime.generate(
            cpp_variant=cpp_model_str,
            cpp_backend=cpp_backend,
            cpp_device=max(0, int(cpp_device or 0)),
            text=str(text),
            reference_audio=str(ref_audio) if ref_audio else None,
            reference_text=str(ref_text or ""),
            top_p=float(top_p),
            top_k=int(top_k),
            temperature=float(temp),
            seed=seed,
            max_tokens=max(1, int(max_tokens)),
            context_length=max(1, int(context_length)),
            text_chunk_size=max(1, int(text_chunk_size)),
            text_chunk_mode=str(text_chunk_mode),
            cpu_expert_layers=max(0, min(40, int(cpu_expert_layers))),
            codec_backend=str(codec_backend),
            cancelled=_inference_cancel_event.is_set,
            multi_reference_cond=multi_reference_cond,
            previous_takes=previous_takes,
            previous_takes_file=previous_takes_file,
            progress=progress,
        )
    except InterruptedError as exc:
        return None, str(exc), selected_seed
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return None, f"Drama 3 C++ synthesis failed: {exc}", selected_seed

    play_done_chime()
    return output_path, "Drama 3 synthesis completed.", selected_seed


def _terminate_cpp_engine():
    was_running = any(
        proc is not None and proc.poll() is None
        for proc in (drama3_runtime._cpp_process, drama3_runtime._speech_process)
    )
    drama3_runtime.shutdown()
    return was_running


def unload_all_models(reason="manual", reset_compiler=True, trim_ram=True):
    _inference_cancel_event.set()
    drama3_runtime.shutdown()
    import gc

    gc.collect()
    if trim_ram:
        _trim_process_ram()
    message = (
        f"[MEMORY] {reason}: Drama 3 runtime processes stopped, GPU memory released, "
        f"and {'RAM working set trimmed' if trim_ram else 'Python garbage collected'}."
    )
    print(message, flush=True)
    return message


def stop_inference():
    _inference_cancel_event.set()
    was_running = _terminate_cpp_engine()
    if was_running:
        return "Drama 3 inference stopped and its runtime was unloaded."
    return "Inference stop requested."

