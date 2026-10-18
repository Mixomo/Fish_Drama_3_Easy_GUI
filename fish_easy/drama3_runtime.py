from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
CPP_MODELS = ROOT / "models" / "Drama-3-GGUF"
OUTPUTS = ROOT / "outputs"
CPP_ROOT = Path(os.environ.get("DRAMA3_AUDIO_CPP_DIR", RUNTIME / "audio.cpp"))
SPEECH_ROOT = Path(os.environ.get("DRAMA3_SPEECH_CPP_DIR", RUNTIME / "speech.cpp"))
CPP_PORT = int(os.environ.get("DRAMA3_CPP_PORT", "3032"))
CPP_URL = f"http://127.0.0.1:{CPP_PORT}"
SPEECH_PORT = int(os.environ.get("DRAMA3_SPEECH_CPP_PORT", "3033"))
SPEECH_URL = f"http://127.0.0.1:{SPEECH_PORT}"
CPP_REPO = "Jmica/Drama-3-GGUF"
SAKASEGAWA_REPO = "sakasegawa/drama-3-GGUF"
SAKASEGAWA_MODEL = "Drama-35B-A3B-v3-Q4_K.gguf"
CPP_VARIANTS = {
    "Q4 experts / Q8 backbone (22.91 GiB) · audio.cpp": (
        "drama-3-backbone-8bit-experts-Q4_K_S.gguf",
        24,
    ),
    "Q3 experts / Q8 backbone (18.93 GiB) · audio.cpp": (
        "drama-3-backbone-8bit-experts-Q3_K_S.gguf",
        12,
    ),
    "Q2 experts / Q8 backbone (15.88 GiB) · audio.cpp": (
        "drama-3-backbone-8bit-experts-Q2_K.gguf",
        8,
    ),
    "Q4 experts / Q4 backbone (21.78 GiB) · audio.cpp": (
        "drama-3-backbone-4bit-experts-Q4_K_S.gguf",
        20,
    ),
    "Drama-35B-A3B-v3-Q4_K (20.79 GB) · speech.cpp": (
        SAKASEGAWA_MODEL,
        0,
    ),
}
CPP_PRESETS = {
    "CUDA": "windows-cuda-release",
    "Vulkan": "windows-vulkan-release",
}
SPEECH_PRESETS = {"Vulkan": "speech.exe"}

_cpp_process: subprocess.Popen | None = None
_cpp_key: tuple | None = None
_cpp_log = None
_speech_process: subprocess.Popen | None = None
_speech_key: tuple | None = None
_speech_log = None
_speech_voice_cache: dict[tuple[str, str, str], Path] = {}


def available_cpp_backends() -> list[str]:
    return [
        backend
        for backend, preset in CPP_PRESETS.items()
        if (
            CPP_ROOT
            / "build"
            / preset
            / "bin"
            / "audiocpp_server.exe"
        ).is_file()
    ]


def available_speech_backends() -> list[str]:
    return ["Vulkan"] if (SPEECH_ROOT / "speech.exe").is_file() else []


def _terminate(process: subprocess.Popen | None, log_file) -> None:
    if process is not None and process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if log_file is not None:
        log_file.close()


def shutdown(clear_voice_cache: bool = False) -> str:
    global _cpp_process, _cpp_key, _cpp_log, _speech_process, _speech_key, _speech_log
    _terminate(_cpp_process, _cpp_log)
    _terminate(_speech_process, _speech_log)
    _cpp_process = None
    _cpp_key = None
    _cpp_log = None
    _speech_process = None
    _speech_key = None
    _speech_log = None
    if clear_voice_cache:
        _speech_voice_cache.clear()
    return "Drama 3 C++ runtimes stopped."
def _cpp_executable(backend: str) -> Path:
    try:
        preset = CPP_PRESETS[backend]
    except KeyError as exc:
        raise ValueError(f"Unknown audio.cpp backend: {backend}") from exc
    candidate = CPP_ROOT / "build" / preset / "bin" / "audiocpp_server.exe"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Drama 3's audio.cpp {backend} server was not found at {candidate}. "
        "Run install.bat to build this backend."
    )


def _wait_for_server(
    process: subprocess.Popen,
    health_url: str,
    log_path: Path,
    cancelled: Callable[[], bool],
) -> None:
    deadline = time.monotonic() + 1800
    try:
        while time.monotonic() < deadline:
            if cancelled():
                raise InterruptedError(
                    "Inference was cancelled while starting Drama 3."
                )
            return_code = process.poll()
            if return_code is not None:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                if return_code == 3221225781:
                    raise RuntimeError(
                        "Drama 3 server exited with Windows status "
                        "0xC0000135 (a required DLL could not be found). "
                        "For CUDA, verify CUDA_PATH and cublas64_13.dll; for "
                        "Vulkan, verify the GPU driver and Vulkan runtime.\n"
                        f"Server log: {log_path}\n{tail}"
                    )
                raise RuntimeError(
                    f"Drama 3 server exited with code {return_code}.\n{tail}"
                )
            try:
                response = requests.get(health_url, timeout=2)
                if response.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError(
            f"Drama 3 server did not become ready at {health_url}. "
            f"See its log: {log_path}"
        )
    except (InterruptedError, TimeoutError):
        _terminate(process, None)
        raise


def _download_cpp_model(variant: str) -> tuple[Path, int]:
    try:
        filename, cpu_expert_layers = CPP_VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown Drama 3 GGUF variant: {variant}") from exc
    if filename == SAKASEGAWA_MODEL:
        model_dir = ROOT / "models" / "Drama-3-speech.cpp"
        model_dir.mkdir(parents=True, exist_ok=True)
        repo_id = SAKASEGAWA_REPO
    else:
        model_dir = CPP_MODELS
        model_dir.mkdir(parents=True, exist_ok=True)
        repo_id = CPP_REPO
    model_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_dir),
        )
    )
    return model_path.resolve(), cpu_expert_layers


def _ensure_cpp_server(
    variant: str,
    backend: str,
    device_index: int,
    context_length: int,
    cpu_expert_layers: int,
    codec_backend: str,
    cancelled: Callable[[], bool],
) -> None:
    global _cpp_process, _cpp_key, _cpp_log

    backend_name = backend.lower()
    if not 1 <= context_length:
        raise ValueError("context_length must be positive.")
    if not 0 <= cpu_expert_layers <= 40:
        raise ValueError("cpu_expert_layers must be between 0 and 40.")
    if codec_backend != "same":
        raise ValueError("Drama 3 codec inference must use the selected GPU backend.")
    if backend_name not in {"cuda", "vulkan"}:
        raise ValueError("Drama 3 requires the CUDA or Vulkan inference backend.")
    model_path, _ = _download_cpp_model(variant)
    executable = _cpp_executable(backend)
    key = (
        str(model_path),
        backend_name,
        str(device_index),
        context_length,
        cpu_expert_layers,
        codec_backend,
    )
    if _cpp_process is not None and _cpp_process.poll() is None:
        if _cpp_key == key:
            return
        _terminate(_cpp_process, _cpp_log)
        _cpp_process = None
        _cpp_log = None

    RUNTIME.mkdir(parents=True, exist_ok=True)
    config_path = RUNTIME / "drama3-audio-cpp-server.json"
    config = {
        "host": "127.0.0.1",
        "port": CPP_PORT,
        "backend": backend_name,
        "threads": max(1, (os.cpu_count() or 4) // 2),
        "lazy_load": True,
        "max_loaded_models": 1,
        "ui": False,
        "models": [
            {
                "id": "drama-3",
                "family": "fish_drama3",
                "path": str(model_path),
                "task": "tts",
                "mode": "streaming",
                "session_options": {
                    "fish_drama3.context_length": context_length,
                },
            }
        ],
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = RUNTIME / "logs" / "drama3-audio-cpp.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_index)
    local_vulkan_bin = RUNTIME / "vulkan" / "bin"
    if local_vulkan_bin.is_dir():
        env["PATH"] = str(local_vulkan_bin) + os.pathsep + env.get("PATH", "")
    if backend_name == "cuda":
        configured_cuda_root = env.get("CUDA_PATH")
        cuda_root = (
            Path(configured_cuda_root).expanduser()
            if configured_cuda_root
            else None
        )
        cuda_bin_dirs = (
            [cuda_root / "bin" / "x64", cuda_root / "bin"]
            if cuda_root
            else []
        )
        for dll_dir in reversed(cuda_bin_dirs):
            if dll_dir.is_dir():
                env["PATH"] = str(dll_dir) + os.pathsep + env.get("PATH", "")
        if os.name == "nt" and not any(
            (dll_dir / "cublas64_13.dll").is_file() for dll_dir in cuda_bin_dirs
        ):
            raise FileNotFoundError(
                "audio.cpp's CUDA server requires cublas64_13.dll, but it was not "
                "found under CUDA_PATH\\bin\\x64 or CUDA_PATH\\bin. "
                f"Current CUDA_PATH: {cuda_root if cuda_root else '(not set)'}. "
                "Install the CUDA 13 runtime/toolkit or choose Vulkan."
            )
    _cpp_log = log_path.open("a", encoding="utf-8")
    try:
        _cpp_process = subprocess.Popen(
            [str(executable), "--config", str(config_path)],
            cwd=CPP_ROOT,
            env=env,
            stdout=_cpp_log,
            stderr=subprocess.STDOUT,
        )
        _cpp_key = key
        _wait_for_server(
            _cpp_process,
            f"{CPP_URL}/health",
            log_path,
            cancelled,
        )
    except Exception:
        _terminate(_cpp_process, _cpp_log)
        _cpp_process = None
        _cpp_key = None
        _cpp_log = None
        raise


def _ensure_speech_server(variant: str, backend: str, device_index: int,
                          cancelled: Callable[[], bool],
                          voice_name: str | None = None,
                          voice_path: Path | None = None) -> None:
    """Start speech.cpp's Drama 3 OpenAI-compatible server."""
    global _speech_process, _speech_key, _speech_log
    backend_name = str(backend).lower()
    if backend_name != "vulkan":
        raise ValueError("speech.cpp 0.8.8 on Windows uses Vulkan; select Vulkan.")
    filename, _ = CPP_VARIANTS[variant]
    if filename != SAKASEGAWA_MODEL:
        raise ValueError("The selected model is not packaged for speech.cpp.")
    global _cpp_process, _cpp_key, _cpp_log
    model_path, _ = _download_cpp_model(variant)
    if _cpp_process is not None and _cpp_process.poll() is None:
        _terminate(_cpp_process, _cpp_log)
        _cpp_process = None
        _cpp_key = None
        _cpp_log = None
    preset = SPEECH_PRESETS[backend]
    candidates = [SPEECH_ROOT / preset,
                  SPEECH_ROOT / "windows-x64-vulkan" / preset]
    executable = next((path for path in candidates if path.is_file()), candidates[0])
    if not executable.is_file():
        raise FileNotFoundError(
            f"speech.cpp {backend} executable not found at {executable}. "
            "Run install.bat to install/build this runtime."
        )
    key = (str(model_path), backend_name, int(device_index),
           str(voice_path.resolve()) if voice_path else None)
    if _speech_process is not None and _speech_process.poll() is None:
        if key == _speech_key:
            return
        _terminate(_speech_process, _speech_log)
        _speech_process = None
        _speech_key = None
        _speech_log = None
    log_path = RUNTIME / "logs" / "drama3-speech-cpp.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _speech_log = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(max(0, int(device_index)))
    if backend_name == "vulkan":
        local_vulkan_bin = RUNTIME / "vulkan" / "bin"
        if local_vulkan_bin.is_dir():
            env["PATH"] = str(local_vulkan_bin) + os.pathsep + env.get("PATH", "")
    command = [str(executable), "serve", "drama-3", "--host", "127.0.0.1",
               "--port", str(SPEECH_PORT), "--device", "gpu"]
    if voice_name and voice_path:
        command.extend(["--add-voice", f"{voice_name}={voice_path.resolve()}"])
    _speech_process = subprocess.Popen(
        command, cwd=SPEECH_ROOT, env=env, stdout=_speech_log,
        stderr=subprocess.STDOUT,
    )
    _speech_key = key
    try:
        _wait_for_server(_speech_process, f"{SPEECH_URL}/health", log_path, cancelled)
    except Exception:
        _terminate(_speech_process, _speech_log)
        _speech_process = None
        _speech_key = None
        _speech_log = None
        raise


def build_request_payload(
    text: str,
    reference_audio: str | None,
    reference_text: str | None,
    top_p: float,
    top_k: int,
    temperature: float,
    seed: int | None,
    max_tokens: int,
    context_length: int,
    text_chunk_size: int,
    text_chunk_mode: str,
    multi_reference_cond: str | None = None,
    previous_takes: str | None = None,
    previous_takes_file: str | None = None,
) -> dict:
    if not text.strip():
        raise ValueError("Text to synthesize cannot be empty.")
    if max_tokens < 1 or context_length < 1 or max_tokens > context_length:
        raise ValueError(
            "max_tokens must be between 1 and the configured context_length."
        )
    if text_chunk_size < 1:
        raise ValueError("text_chunk_size must be positive.")
    if not 0 < top_p <= 1 or top_k < 1 or not 0 < temperature < 2:
        raise ValueError("Invalid Drama 3 sampling settings.")
    if text_chunk_mode not in {"default", "tag_aware", "japanese", "endline"}:
        raise ValueError(f"Unsupported text_chunk_mode: {text_chunk_mode}")
    if reference_audio and multi_reference_cond:
        raise ValueError(
            "Use either the single Reference Audio control or the multi-reference JSON, not both."
        )

    payload = {
        "model": "drama-3",
        "input": text,
        "response_format": "wav",
        "top_p": float(top_p),
        "top_k": int(top_k),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "options": {
            "text_chunk_size": str(int(text_chunk_size)),
            "text_chunk_mode": text_chunk_mode,
        },
    }
    for option_name, option_value, expected_type in (
        ("multi_reference_cond", multi_reference_cond, list),
        ("previous_takes", previous_takes, (list, str)),
    ):
        if option_value and option_value.strip():
            try:
                parsed = json.loads(option_value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{option_name} must contain valid JSON."
                ) from exc
            if not isinstance(parsed, expected_type):
                raise ValueError(
                    f"{option_name} must be JSON of type "
                    f"{'array' if expected_type is list else 'array or string'}."
                )
            payload["options"][option_name] = option_value
    if previous_takes_file:
        history_path = Path(previous_takes_file).expanduser().resolve()
        if not history_path.is_file():
            raise FileNotFoundError(
                f"Previous-takes JSON file was not found: {history_path}"
            )
        if previous_takes and previous_takes.strip():
            raise ValueError(
                "Use either Previous takes JSON or Previous takes file, not both."
            )
        payload["options"]["previous_takes_file"] = str(history_path)
    if seed is not None:
        payload["seed"] = int(seed)
    if reference_audio:
        if not reference_text or not reference_text.strip():
            raise ValueError(
                "The reference transcript is required when reference audio is provided."
            )
        reference_path = Path(reference_audio).expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference audio was not found: {reference_path}")
        payload["voice_ref"] = str(reference_path)
        payload["options"]["reference_text"] = reference_text
    return payload


def generate(
    cpp_variant: str,
    cpp_backend: str,
    cpp_device: int,
    text: str,
    reference_audio: str | None,
    reference_text: str | None,
    top_p: float,
    top_k: int,
    temperature: float,
    seed: int | None,
    max_tokens: int,
    context_length: int,
    text_chunk_size: int,
    text_chunk_mode: str,
    cpu_expert_layers: int,
    codec_backend: str,
    multi_reference_cond: str | None,
    previous_takes: str | None,
    previous_takes_file: str | None,
    cancelled: Callable[[], bool],
    progress: Callable,
) -> tuple[str, int]:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / f"drama3_{int(time.time() * 1000)}.wav"
    filename, _ = CPP_VARIANTS[cpp_variant]
    if filename == SAKASEGAWA_MODEL:
        model_path, _ = _download_cpp_model(cpp_variant)
        executable = next((SPEECH_ROOT / candidate for candidate in
                           ("speech.exe", "windows-x64-vulkan/speech.exe")
                           if (SPEECH_ROOT / candidate).is_file()),
                          SPEECH_ROOT / "speech.exe")
        payload = {
            "model": "drama-3",
            "input": text,
            "voice": "none",
            "response_format": "wav",
            "top_p": float(top_p),
            "top_k": int(top_k),
            "temperature": float(temperature),
            "experts_on_cpu": False,
            "seed": int(seed) if seed is not None else None,
            "max_seconds": min(95.1, max(0.1, float(max_tokens) / 21.5)),
        }
        voice_name = None
        voice_path = None
        if reference_audio:
            reference_path = Path(reference_audio).expanduser().resolve()
            if not reference_path.is_file() or not str(reference_text or "").strip():
                raise ValueError("speech.cpp voice cloning requires reference audio and its transcript.")
            cache_key = (str(model_path), str(reference_path), str(reference_text))
            voice_path = _speech_voice_cache.get(cache_key)
            voice_name = "reference"
            if voice_path is None or not voice_path.is_file():
                voice_dir = ROOT / "models" / "Drama-3-speech.cpp" / "voices"
                voice_dir.mkdir(parents=True, exist_ok=True)
                voice_path = voice_dir / f"voice_{abs(hash(cache_key)):x}.gguf"
                make_voice = [str(executable), "voice", "drama-3", str(reference_path),
                              "--transcript", str(reference_text), "--device", "gpu", str(voice_path)]
                result = subprocess.run(make_voice, cwd=SPEECH_ROOT, capture_output=True,
                                        text=True, timeout=1800)
                if result.returncode or not voice_path.is_file():
                    raise RuntimeError(f"speech.cpp could not encode the reference: {result.stderr[-3000:]}")
                _speech_voice_cache[cache_key] = voice_path
            payload["voice"] = voice_name
        if multi_reference_cond:
            raise ValueError("speech.cpp does not accept the audio.cpp multi-reference option.")
        if previous_takes or previous_takes_file:
            raise ValueError("speech.cpp does not support previous-take continuation.")
        _ensure_speech_server(cpp_variant, cpp_backend, cpp_device, cancelled,
                              voice_name=voice_name, voice_path=voice_path)
        progress(0.45, desc=f"Generating with Drama 3 speech.cpp {cpp_backend}")
        try:
            response = requests.post(f"{SPEECH_URL}/v1/audio/speech", json=payload,
                                 timeout=(15, 1800))
        except requests.RequestException as exc:
            raise RuntimeError(f"speech.cpp server request failed: {exc}") from exc
        if not response.ok:
            raise RuntimeError(
                f"speech.cpp request failed ({response.status_code}): {response.text[:4000]}"
            )
        if not response.content.startswith(b"RIFF"):
            raise RuntimeError("speech.cpp returned a response that is not a WAV file.")
        out_path.write_bytes(response.content)
        sample_rate = int(response.headers.get("X-Sample-Rate", "44100"))
        return str(out_path), sample_rate
    progress(0.1, desc=f"Downloading/checking {cpp_variant}")
    _ensure_cpp_server(
        cpp_variant,
        cpp_backend,
        cpp_device,
        context_length,
        cpu_expert_layers,
        codec_backend,
        cancelled,
    )
    payload = build_request_payload(
        text=text,
        reference_audio=reference_audio,
        reference_text=reference_text,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        context_length=context_length,
        text_chunk_size=text_chunk_size,
        text_chunk_mode=text_chunk_mode,
        multi_reference_cond=multi_reference_cond,
        previous_takes=previous_takes,
        previous_takes_file=previous_takes_file,
    )

    progress(0.45, desc=f"Generating with Drama 3 {cpp_backend}")
    try:
        response = requests.post(
            f"{CPP_URL}/v1/audio/speech",
            json=payload,
            timeout=(15, 1800),
        )
    except requests.RequestException as exc:
        if cancelled():
            raise InterruptedError("Drama 3 C++ inference was cancelled.") from exc
        raise
    if not response.ok:
        message = response.text[:4000]
        if (
            response.status_code == 400
            and "prompt plus max_tokens exceeds" in message
        ):
            message += (
                "\nReduce Max acoustic frames, shorten the input/reference audio "
                "or transcript, or increase Context length. The context includes "
                "the prompt, reference frames, text history, and generated frames."
            )
        raise RuntimeError(
            f"audio.cpp request failed ({response.status_code}): {message}"
        )
    if not response.content.startswith(b"RIFF"):
        raise RuntimeError(
            "audio.cpp returned a response that is not a WAV file "
            f"(content-type: {response.headers.get('content-type', 'unknown')})."
        )
    out_path.write_bytes(response.content)
    return str(out_path), 44100








