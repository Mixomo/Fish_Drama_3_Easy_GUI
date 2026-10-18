from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import gradio as gr

import fish_backend as B
from fish_easy.console import html_view, log
APP_TITLE = "Fish Audio Drama 3 Easy GUI"
NONE = "None"
MAX_DIALOGUE_SEGMENTS = 20

# Copied structurally from the newest CosyVoice3 Easy GUI. Native Gradio theme
# owns colors/fonts; this CSS only enforces family hierarchy, spacing and cards.
CSS = """
html, body, #root { width:100%; min-width:0; }
.gradio-container { max-width:none !important; width:100% !important; margin:0 auto !important; padding:18px 28px 28px !important; }
.tabs, .tabitem, .gradio-container .block, .gradio-container .form { width:100% !important; max-width:none !important; }
.title-section { border-bottom:1px solid var(--border-color-primary); margin-bottom:8px; padding-bottom:8px; align-items:center !important; }
.title-section .prose, .title-section h1 { margin:0 !important; padding:0 !important; }
.title-section button { min-height:36px !important; white-space:nowrap; }
.tab-subtitle { opacity:.78; margin:0 0 12px 0 !important; padding:0 !important; }
.section-heading { margin:10px 0 2px !important; }
.runtime-row, .workflow-actions, .dialogue-toolbar { gap:10px !important; align-items:end !important; }
.compact-button { min-width:52px !important; max-width:110px !important; }
.medium-control { max-width:420px !important; }
.small-control { max-width:260px !important; }
.dialogue-turn-card { padding:14px 14px 12px !important; margin:6px 0 12px !important; border:1px solid var(--border-color-primary) !important; border-radius:10px !important; background:transparent !important; box-shadow:none !important; }
.dialogue-turn-title { margin:0 0 6px 0 !important; opacity:.86; }
.dialogue-actions { gap:7px !important; align-items:center !important; }
.dialogue-actions button { min-width:88px !important; padding-left:10px !important; padding-right:10px !important; white-space:nowrap !important; }
.audio-safe-space { overflow:visible !important; padding-bottom:18px !important; border:0 !important; box-shadow:none !important; }
.output-clean, .output-clean > div, .output-clean .wrap { border:0 !important; box-shadow:none !important; }
.compact-status { margin:0 0 8px 0 !important; padding:0 !important; min-height:0 !important; }
.console-accordion, .console-accordion > div { border-radius:8px !important; }
.voice-library-row { align-items:end !important; gap:8px !important; }
.voice-library-row button { min-width:48px !important; max-width:52px !important; }
footer { display:none !important; }
@media(max-width:900px){ .gradio-container{padding:10px!important}.title-section button{min-width:100% !important;} .medium-control,.small-control{max-width:none!important;} }
"""


def _whisper_model(display): return B.WHISPER_MODELS.get(display, display)

def _voice_choices(): return [NONE, *B.get_sample_choices()]

def _transcribe(audio, model_display, batch_size=4):
    # Fish synthesis is language-free and the Easy GUI keeps ASR equally simple:
    # Faster-Whisper auto-detects instead of exposing another language dropdown.
    return B.transcribe_only(audio, _whisper_model(model_display), "Auto-detect", batch_size)


def _clear_vram():
    return B.unload_all_models(
        reason="manual Unload All Models / Free RAM - VRAM",
        reset_compiler=True,
        trim_ram=True,
    )


def _clear_outputs():
    root=Path(B.OUTPUTS_DIR); root.mkdir(parents=True, exist_ok=True); removed=0
    for item in root.iterdir():
        try:
            shutil.rmtree(item) if item.is_dir() else item.unlink(); removed += 1
        except Exception: pass
    return f"Removed {removed} generated output item(s)."


def _inference_ui_start():
    B.begin_inference()
    return gr.update(interactive=False), gr.update(interactive=True), "⏳ Inference running…"


def _inference_ui_finish(status):
    return gr.update(interactive=True), gr.update(interactive=False), status


def _stop_inference_ui():
    status = B.stop_inference()
    return gr.update(interactive=True), gr.update(interactive=False), status


def _resolve_inference_seed(seed, fixed_seed):
    # Unlocked = a fresh random seed for every Generate click, regardless of
    # the value left visible in the Seed box by the previous generation.
    # Locked = preserve/use the visible manual or previously generated seed.
    if bool(fixed_seed):
        chosen = B.normalize_seed(seed)
        return chosen if chosen is not None else B.random_seed_value()
    return B.random_seed_value()


def _run_single(
    cpp_model,
    cpp_backend,
    cpp_device,
    text,
    ref_audio,
    ref_text,
    top_p,
    top_k,
    temp,
    max_tokens,
    context_length,
    text_chunk_size,
    text_chunk_mode,
    cpu_expert_layers,
    codec_backend,
    chunk_mode,
    chunk_gap,
    multi_reference_cond,
    previous_takes,
    previous_takes_file,
    seed,
    fixed_seed,
):
    chosen=_resolve_inference_seed(seed, fixed_seed)
    wav,status,_=B.clone_voice(
        cpp_model,
        cpp_backend,
        cpp_device,
        text,
        ref_audio,
        ref_text,
        top_p,
        top_k,
        temp,
        chosen,
        max_tokens=max_tokens,
        context_length=context_length,
        text_chunk_size=text_chunk_size,
        text_chunk_mode=text_chunk_mode,
        cpu_expert_layers=cpu_expert_layers,
        codec_backend=codec_backend,
        chunk_mode=chunk_mode,
        chunk_gap=chunk_gap,
        multi_reference_cond=multi_reference_cond,
        previous_takes=previous_takes,
        previous_takes_file=previous_takes_file,
    )
    return wav,status,chosen,chosen

def _run_dialogue(
    cpp_model,
    cpp_backend,
    cpp_device,
    top_p,
    top_k,
    temp,
    max_tokens,
    context_length,
    text_chunk_size,
    text_chunk_mode,
    cpu_expert_layers,
    codec_backend,
    chunk_mode,
    chunk_gap,
    multi_reference_cond,
    previous_takes,
    previous_takes_file,
    seed,
    fixed_seed,
    row_count,
    silence,
    *args,
):
    chosen=_resolve_inference_seed(seed, fixed_seed)
    if cpp_model.endswith("speech.cpp"):
        multi_reference_cond = None
        previous_takes = None
        previous_takes_file = None
    elif multi_reference_cond and multi_reference_cond.strip():
        message = (
            "Multiple-reference JSON is for Single Inference. In Dialogue Builder, "
            "assign a saved voice separately to each turn."
        )
        return None,message,chosen,chosen
    dialogue_args = list(args)
    sample_names = dialogue_args[:MAX_DIALOGUE_SEGMENTS]
    turn_texts = dialogue_args[MAX_DIALOGUE_SEGMENTS:]
    wav,status,_=B.generate_dialogue(
        cpp_model,
        cpp_backend,
        cpp_device,
        top_p,
        top_k,
        temp,
        chosen,
        row_count,
        silence,
        *dialogue_args,
        max_tokens=max_tokens,
        context_length=context_length,
        text_chunk_size=text_chunk_size,
        text_chunk_mode=text_chunk_mode,
        cpu_expert_layers=cpu_expert_layers,
        codec_backend=codec_backend,
        chunk_mode=chunk_mode,
        chunk_gap=chunk_gap,
        multi_reference_cond=multi_reference_cond,
        previous_takes=previous_takes,
        previous_takes_file=previous_takes_file,
    )
    return wav,status,chosen,chosen

def _dialogue_refresh(*current):
    choices=_voice_choices()
    return [gr.update(choices=choices, value=(v if v in choices else NONE)) for v in current]

def _default_cpu_expert_layers(model):
    return gr.update(value=B.GGUF_MODELS.get(model, ("", 24))[1])


def _runtime_backend_for_model(model, current_backend):
    if str(model or "").endswith("speech.cpp"):
        choices = list(B.SPEECH_BACKENDS)
    else:
        choices = list(B.CPP_BACKENDS)
    value = current_backend if current_backend in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def _dialogue_mutate(action, index, count, *args):
    samples=list(args[:MAX_DIALOGUE_SEGMENTS]); texts=list(args[MAX_DIALOGUE_SEGMENTS:]); count=int(count)
    if action == "insert" and count < MAX_DIALOGUE_SEGMENTS:
        samples.insert(index+1, samples[index]); texts.insert(index+1, ""); samples.pop(); texts.pop(); count += 1
    elif action == "clone" and count < MAX_DIALOGUE_SEGMENTS:
        samples.insert(index+1, samples[index]); texts.insert(index+1, texts[index]); samples.pop(); texts.pop(); count += 1
    elif action == "remove" and count > 1:
        samples.pop(index); texts.pop(index); samples.append(NONE); texts.append(""); count -= 1
    return [count] + [gr.update(value=samples[i], visible=i<count) for i in range(MAX_DIALOGUE_SEGMENTS)] + [gr.update(value=texts[i], visible=i<count) for i in range(MAX_DIALOGUE_SEGMENTS)] + [gr.update(visible=i<count) for i in range(MAX_DIALOGUE_SEGMENTS)]


def _new_seed_pair():
    value=B.random_seed_value(); return value,value


_voice_init=_voice_choices()

with gr.Blocks(title=APP_TITLE, css=CSS, theme=gr.themes.Default()) as app:
    with gr.Row(elem_classes="title-section"):
        with gr.Column(scale=7, min_width=420):
            gr.Markdown("# 🎙️ Fish Audio Drama 3 Easy GUI")
            gr.Markdown(
                "Drama 3 · audio.cpp / speech.cpp GPU inference · reusable Voice Library · single and multi-speaker synthesis",
                elem_classes="tab-subtitle",
            )
        with gr.Column(scale=3, min_width=480):
            with gr.Row():
                unload_btn=gr.Button("🧹 Unload All Models / Free RAM - VRAM", variant="secondary", size="sm")
                clear_outputs_btn=gr.Button("🗑️ Clear Outputs", variant="secondary", size="sm")
    hidden_status=gr.State("")
    unload_btn.click(_clear_vram, outputs=hidden_status, queue=False)
    clear_outputs_btn.click(_clear_outputs, outputs=hidden_status, queue=False)

    with gr.Tabs(elem_classes="tabs"):
        # ------------------------------------------------------------------
        # PREP SAMPLES
        # ------------------------------------------------------------------
        with gr.Tab("🎙️ Prep Samples"):
            gr.Markdown("*Create reusable reference voices for Single Inference and Dialogue Builder.*", elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide", open=False):
                gr.Markdown("""### Workflow
1. **Load or record a reference** — use a clean, dry WAV clip. The official model-card example is 3.9 seconds; no hard reference-duration limit is published.
2. **Reference Transcript** — enter exactly what is spoken. The reference consumes part of the model context, so concise clips leave more room for output.
3. **Faster-Whisper (optional)** — use it only to bootstrap the transcript. Language detection is always automatic; review the transcript before saving.
4. **Voice Name → Save Voice** — stores audio + transcript in the shared Voice Library. Choosing **None** clears the saved selection without deleting anything.
5. **↻ / 🗑️** — refresh the library after external file changes or delete the currently selected saved voice.

### What happens internally
Drama 3 handles reference conditioning. Saving a voice does not alter model weights; it creates a reusable reference pair.""")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row(elem_classes="voice-library-row"):
                        prep_voice=gr.Dropdown(_voice_init, value=NONE, label="Voice Library", scale=8,
                                               info="Load a saved reference pair. Select None to clear the current library selection.")
                        prep_refresh=gr.Button("↻", size="sm", variant="secondary")
                        prep_delete=gr.Button("🗑️", size="sm", variant="stop")
                    prep_audio=gr.Audio(type="filepath", label="Reference Audio", sources=["upload","microphone"], elem_classes="audio-safe-space")
                    prep_text=gr.Textbox(label="Reference Transcript", lines=5,
                                         info="Exact transcript of the reference clip; used for voice cloning/prompt conditioning.")
                    with gr.Accordion("🛰️ Faster-Whisper Transcription", open=False):
                        prep_wm=gr.Dropdown(list(B.WHISPER_MODELS), value="large-v3 (~10 GB VRAM)", label="Whisper Model",
                                            info="ASR model used only to transcribe the reference; larger models are more accurate and use more VRAM.")
                        prep_wb=gr.Slider(1,16,4,1,label="ASR Batch Size",
                                          info="Faster-Whisper batch size. Increase for speed when VRAM allows; lower it if transcription runs out of memory.")
                        with gr.Row():
                            prep_transcribe=gr.Button("Transcribe Reference", variant="secondary", scale=5)
                            prep_transcribe_stop=gr.Button("🛑 Stop", variant="stop", scale=1)
                with gr.Column(scale=1):
                    prep_name=gr.Textbox(label="Voice Name", placeholder="voice_name",
                                         info="Library name used by inference and Dialogue Builder.")
                    prep_save=gr.Button("Save Voice", variant="primary", size="lg")
                    prep_status=gr.Markdown("No saved voice selected.", elem_classes="compact-status")
            prep_voice.change(_sample_payload, prep_voice, [prep_audio,prep_text,prep_status], queue=False)
            prep_refresh.click(_refresh_voice, prep_voice, prep_voice, queue=False)
            prep_delete_event=prep_delete.click(_delete_voice, prep_voice, [prep_voice,prep_audio,prep_text,prep_status], queue=False)
            prep_transcribe_event=prep_transcribe.click(_transcribe, [prep_audio,prep_wm,prep_wb], prep_text)
            prep_transcribe_stop.click(B.stop_aux_job, outputs=hidden_status, cancels=[prep_transcribe_event], queue=False)
            prep_save_event=prep_save.click(_save_voice, [prep_audio,prep_name,prep_text], [prep_status,prep_voice], queue=False)

        # ------------------------------------------------------------------
        # INFERENCE
        # ------------------------------------------------------------------
        with gr.Tab("🔊 Inference"):
            gr.Markdown("*Native GPU inference with audio.cpp and speech.cpp; Single Inference and Dialogue Builder share the Voice Library.*", elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide", open=False):
                gr.Markdown("""### Runtime controls
- **GGUF / Compute Backend / GPU Device** — choose a Drama 3 GGUF and a compiled CUDA or Vulkan runtime.
- **Unload All Models / Free RAM - VRAM** stops the active C++ server and releases reclaimable CUDA and RAM state.

### Generation controls
`Max acoustic frames` is the per-text-chunk generation cap (about 21.5 frames/second), not a seconds limit. `Context length` counts the prompt, reference audio, earlier takes, and generated frames; increasing it may use more memory. If the server reports that prompt plus max_tokens exceeds context, reduce the frame cap, shorten the prompt/reference, or increase context. Top-P, Top-K, Temperature, Text chunk size/mode, and Seed map to audio.cpp request options.

**Chunking Rule / Chunk Size / Chunk Gap** configure long-form segmentation. **Advanced references / continuation** accepts multiple voice references and previous-take history as JSON or a JSON file when using audio.cpp.

### Long-form workflow
Drama 3 supports native chunking; optional GUI chunking can split long text into configurable pieces with a gap. The reference audio API has no separate fixed maximum duration: longer references consume more context. **Dialogue Silence** controls the gap between turns generated in Dialogue Builder.

### Voice conditioning
Reference audio is optional. For voice cloning, use a clean WAV and its exact transcript. See the [official Drama 3 model card](https://huggingface.co/fishaudio/drama-3) for examples.""")
            gr.Markdown("### ⚙️ Runtime", elem_classes="section-heading")
            with gr.Row(elem_classes="runtime-row"):
                cpp_model=gr.Dropdown(list(B.GGUF_MODELS), value=list(B.GGUF_MODELS)[0], label="GGUF Model", scale=3,
                                      info="Select an audio.cpp or speech.cpp Drama 3 GGUF.")
                cpp_backend=gr.Dropdown(list(B.CPP_BACKENDS), value=B.CPP_BACKENDS[0] if B.CPP_BACKENDS else None, label="Compute Backend", scale=2,
                                        info="Select a compiled CUDA or Vulkan GPU backend.")
                cpp_device=gr.Number(0, precision=0, minimum=0, label="GPU Device", scale=1,
                                     info="Zero-based CUDA/Vulkan GPU index.")
            with gr.Accordion("🎛️ Generation Settings", open=True):
                with gr.Row():
                    max_tokens=gr.Slider(1,4096,500,25,label="Max acoustic frames",info="Maximum generated frames per text chunk (~21.5 frames/s). This is not a hard seconds limit.")
                    context_length=gr.Slider(512,8192,4096,256,label="Context length (frames)",info="Capacity for prompt + reference + history + generated frames. Larger contexts can use more memory; restart occurs when changed.")
                    text_chunk_size=gr.Slider(25,1000,200,25,label="Chunk size",info="Maximum characters per GUI text chunk.")
                    text_chunk_mode=gr.Dropdown(["tag_aware","default","japanese","endline"],value="tag_aware",label="audio.cpp chunk mode",visible=False)
                with gr.Row():
                    top_p=gr.Slider(.01,1,.9,.01,label="Top-P",info="Nucleus sampling cutoff; valid range is (0, 1].")
                    top_k=gr.Slider(1,100,30,1,label="Top-K",info="Number of candidate tokens retained for sampling.")
                    temperature=gr.Slider(.05,1.95,1.0,.05,label="Temperature",info="Sampling temperature; audio.cpp accepts values strictly between 0 and 2.")
                with gr.Row():
                    seed=gr.Number(0,precision=0,minimum=0,maximum=(1<<32)-1,label="Seed",info="Unsigned 32-bit sampling seed. It is only kept for the next generation when Enable Fixed Seed is ON.")
                    fixed_seed=gr.Checkbox(False,label="Enable Fixed Seed",info="OFF = new random seed on every Generate. ON = preserve and reuse the visible Seed; manual seeds remain untouched.")
                    last_seed=gr.State(0)
                    reuse_seed=gr.Button("Reuse Last Seed",size="sm")
                    random_seed=gr.Button("Random Seed",size="sm")
                    dialogue_silence=gr.Slider(0,5,.5,.1,label="Dialogue Silence (s)",info="Silence inserted between Dialogue Builder turns only.")
                reuse_seed.click(lambda s:s,last_seed,seed,queue=False)
                random_seed.click(_new_seed_pair,outputs=[seed,last_seed],queue=False)
            with gr.Accordion("🛠️ Advanced runtime settings", open=False):
                with gr.Row():
                    cpu_expert_layers=gr.Slider(0,40,24,1,label="audio.cpp expert offload",info="audio.cpp only: routed experts stored/evaluated in system RAM.",visible=False, interactive=False)
                    codec_backend=gr.Dropdown(["same"],value="same",label="audio.cpp codec backend",info="Drama 3 inference runs on the selected GPU backend.",visible=False, interactive=False)
                with gr.Row():
                    chunk_mode=gr.Dropdown(B.CHUNK_CHOICES,value="None",label="Chunking Rule",info="Split long-form text before each runtime request.")
                    chunk_gap=gr.Slider(0,5,0.3,0.1,label="Chunk Gap (s)",info="Silence inserted between generated chunks.")
                cpp_model.change(
                    _default_cpu_expert_layers,
                    cpp_model,
                    cpu_expert_layers,
                    queue=False,
                )
                cpp_model.change(
                    _runtime_backend_for_model,
                    [cpp_model, cpp_backend],
                    cpp_backend,
                    queue=False,
                )
            with gr.Accordion("🧩 Advanced references / continuation", open=False):
                multi_reference_cond=gr.Textbox(
                    label="Multiple voice references (JSON)",
                    lines=4,
                    placeholder='[{"text":"<|speaker:0|> Hello there.","audio_path":"C:\\\\voices\\\\speaker0.wav"}]',
                    info="Single Inference only. Dialogue Builder assigns saved voices per turn; do not combine this with a selected reference.",
                )
                previous_takes=gr.Textbox(
                    label="Previous takes (JSON)",
                    lines=3,
                    placeholder='["The original line spoken earlier."]',
                    info="Optional JSON array/string of earlier takes for continuation or editing. It is included in the context.",
                )
                previous_takes_file=gr.File(
                    label="Previous takes JSON file (optional; alternative to the JSON field)",
                    file_types=[".json"],
                    type="filepath",
                )
            with gr.Accordion("❓ Pronunciation / Control Markup Guide", open=False):
                gr.Markdown("""### Event and delivery cues
Place bracketed event cues where they should occur in the utterance. The official model card demonstrates `[sighs]` and `[laughs]`:

```text
[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair.
```

The cues are part of the text prompt; delivery is model-generated and may vary between takes.

### Multiple characters
Use speaker markers `<|speaker:0|>` through `<|speaker:15|>` to label up to 16 character turns:

```text
<|speaker:0|> Are we ready?
<|speaker:1|> Almost. Give me a minute.
<|speaker:0|> All right, let's go.
```

For speaker-conditioned output, provide the matching voice reference(s). This GUI's single-reference field conditions one voice; Dialogue Builder supports a separate saved voice per turn.

### Reference length and context
The audio.cpp port describes `max_tokens` as acoustic frames (about 21.5 per second) and context as prompt plus generated frames. The model card does not specify a hard maximum in seconds/minutes for reference audio. Keep reference audio concise, provide its exact transcript, and leave room in **Context length** for the requested output.""")

            with gr.Tabs():
                with gr.Tab("Single Inference"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row(elem_classes="voice-library-row"):
                                infer_voice=gr.Dropdown(_voice_init,value=NONE,label="Voice Library",scale=8,
                                                        info="Select a reusable reference voice, or None for unconditioned generation.")
                                infer_refresh=gr.Button("↻",size="sm",variant="secondary")
                            infer_audio=gr.Audio(type="filepath",label="Reference Audio (Optional; no fixed duration cap published)",sources=["upload"],elem_classes="audio-safe-space")
                            infer_ref_text=gr.Textbox(label="Reference Transcript (Required only with reference audio)",lines=4,
                                                      info="Exact transcript of the reference audio. Filled automatically when a saved voice is loaded.")
                            infer_voice_status=gr.Markdown("No saved voice selected.",elem_classes="compact-status")
                            with gr.Accordion("🛰️ Transcribe Uploaded Reference",open=False):
                                infer_wm=gr.Dropdown(list(B.WHISPER_MODELS),value="large-v3 (~10 GB VRAM)",label="Whisper Model",
                                                     info="Optional Faster-Whisper model used only for reference transcription.")
                                infer_wb=gr.Slider(1,16,4,1,label="ASR Batch Size",
                                                  info="Faster-Whisper batch size used only for uploaded-reference transcription.")
                                with gr.Row():
                                    infer_transcribe=gr.Button("Transcribe",variant="secondary",scale=5)
                                    infer_transcribe_stop=gr.Button("🛑 Stop",variant="stop",scale=1)
                        with gr.Column(scale=2):
                            target_text=gr.Textbox(label="Text to Generate",lines=8,
                                                   info="Target text. Language is inferred automatically by Drama 3.")
                            with gr.Row():
                                generate=gr.Button("Generate Audio 🚀",variant="primary",size="lg",scale=5)
                                stop_generate=gr.Button("🛑 Stop",variant="stop",size="lg",interactive=False,scale=1)
                            output_audio=gr.Audio(label="Generated Audio",type="filepath",elem_classes="output-clean")
                            infer_status=gr.Markdown("")
                    infer_voice.change(_sample_payload,infer_voice,[infer_audio,infer_ref_text,infer_voice_status],queue=False)
                    infer_refresh.click(_refresh_voice,infer_voice,infer_voice,queue=False)
                    infer_transcribe_event=infer_transcribe.click(_transcribe,[infer_audio,infer_wm,infer_wb],infer_ref_text)
                    infer_transcribe_stop.click(B.stop_aux_job,outputs=hidden_status,cancels=[infer_transcribe_event],queue=False)
                    _single_start = generate.click(
                        _inference_ui_start, None, [generate, stop_generate, infer_status], queue=False
                    )
                    _single_run = _single_start.then(
                        _run_single,
                        [cpp_model,cpp_backend,cpp_device,target_text,infer_audio,infer_ref_text,top_p,top_k,temperature,max_tokens,context_length,text_chunk_size,text_chunk_mode,cpu_expert_layers,codec_backend,chunk_mode,chunk_gap,multi_reference_cond,previous_takes,previous_takes_file,seed,fixed_seed],
                        [output_audio,infer_status,seed,last_seed],
                    )
                    _single_run.then(
                        _inference_ui_finish, infer_status, [generate, stop_generate, infer_status], queue=False
                    )
                    stop_generate.click(
                        _stop_inference_ui, None, [generate, stop_generate, infer_status],
                        cancels=[_single_run], queue=False
                    )

                with gr.Tab("Dialogue Builder"):
                    gr.Markdown("### 💬 Multi-Speaker Dialogue Builder",elem_classes="section-heading")
                    gr.Markdown("Each turn selects a Voice Library reference and its own text. Insert/Clone/Remove follows the same row workflow as the sister GUIs; global generation settings above apply to every turn.", elem_classes="tab-subtitle")
                    with gr.Row(elem_classes="dialogue-toolbar"):
                        dialogue_voice_refresh=gr.Button("↻ Refresh Voice Library",variant="secondary")
                    dialogue_count=gr.State(2); d_groups=[]; d_voices=[]; d_texts=[]; d_insert=[]; d_clone=[]; d_remove=[]
                    for i in range(MAX_DIALOGUE_SEGMENTS):
                        with gr.Group(visible=i<2,elem_classes="dialogue-turn-card") as group:
                            gr.Markdown(f"**Turn {i+1}**",elem_classes="dialogue-turn-title")
                            with gr.Row():
                                voice=gr.Dropdown(_voice_init,value=NONE,label="Voice",scale=2,
                                                  info="Saved reference voice for this dialogue turn.")
                                text=gr.Textbox(label="Text",lines=3,scale=6,
                                                info="Text spoken by this turn; Fish detects language automatically.")
                            with gr.Row(elem_classes="dialogue-actions"):
                                ins=gr.Button("＋ Insert",size="sm"); clone=gr.Button("⧉ Clone",size="sm"); rem=gr.Button("− Remove",size="sm")
                        d_groups.append(group); d_voices.append(voice); d_texts.append(text); d_insert.append(ins); d_clone.append(clone); d_remove.append(rem)
                    inputs=[dialogue_count,*d_voices,*d_texts]; outputs=[dialogue_count,*d_voices,*d_texts,*d_groups]
                    for i in range(MAX_DIALOGUE_SEGMENTS):
                        d_insert[i].click(lambda count,*vals,_i=i: _dialogue_mutate("insert",_i,count,*vals), inputs, outputs, queue=False)
                        d_clone[i].click(lambda count,*vals,_i=i: _dialogue_mutate("clone",_i,count,*vals), inputs, outputs, queue=False)
                        d_remove[i].click(lambda count,*vals,_i=i: _dialogue_mutate("remove",_i,count,*vals), inputs, outputs, queue=False)
                    dialogue_voice_refresh.click(_dialogue_refresh,d_voices,d_voices,queue=False)
                    with gr.Row():
                        dialogue_generate=gr.Button("Generate Dialogue 🚀",variant="primary",size="lg",scale=5)
                        dialogue_stop=gr.Button("🛑 Stop",variant="stop",size="lg",scale=1)
                    dialogue_output=gr.Audio(label="Dialogue Output",type="filepath",elem_classes="output-clean")
                    dialogue_status=gr.Markdown("")
                    gr.Markdown("Each filled turn runs as a separate inference. The model runtime is unloaded and memory is reclaimed before the next turn.")
                    _dialogue_start=dialogue_generate.click(_inference_ui_start,None,[dialogue_generate,dialogue_stop,dialogue_status],queue=False)
                    _dialogue_run=_dialogue_start.then(
                        _run_dialogue,
                        [cpp_model,cpp_backend,cpp_device,top_p,top_k,temperature,max_tokens,context_length,text_chunk_size,text_chunk_mode,cpu_expert_layers,codec_backend,chunk_mode,chunk_gap,multi_reference_cond,previous_takes,previous_takes_file,seed,fixed_seed,dialogue_count,dialogue_silence,*d_voices,*d_texts],
                        [dialogue_output,dialogue_status,seed,last_seed],
                    )
                    _dialogue_run.then(_inference_ui_finish,dialogue_status,[dialogue_generate,dialogue_stop,dialogue_status],queue=False)
                    dialogue_stop.click(_stop_inference_ui,None,[dialogue_generate,dialogue_stop,dialogue_status],cancels=[_dialogue_run],queue=False)

    prep_save_event.then(lambda: [_refresh_voice(NONE), *[gr.update(choices=_voice_choices(), value=NONE) for _ in range(MAX_DIALOGUE_SEGMENTS)]], outputs=[infer_voice,*d_voices],queue=False)
    prep_delete_event.then(lambda: [_refresh_voice(NONE), *[gr.update(choices=_voice_choices(), value=NONE) for _ in range(MAX_DIALOGUE_SEGMENTS)]], outputs=[infer_voice,*d_voices],queue=False)

    # One shared, self-contained console below every tab — same renderer as CosyVoice3.
    with gr.Accordion("🖥️ Console", open=True, elem_classes="console-accordion"):
        console=gr.HTML(value=html_view(), elem_id="drama3-console")
        console_timer=gr.Timer(0.5)
        console_timer.tick(html_view,outputs=console,queue=False,show_progress="hidden")

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
