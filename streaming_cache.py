#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streaming TTS demo that prebuilds and reuses the prompt cache.
Useful for lowering first-audio latency when repeatedly synthesizing
with the same reference voice.
"""

import time
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
from voxcpm import VoxCPM
from voxcpm.utils.text_normalize import TextNormalizer

SAMPLE_RATE = 16000


def stream_text(model: VoxCPM, prompt_cache: dict, text: str, idx: int, inference_steps: int = 10):
    """Stream one text utterance using a prebuilt prompt cache."""
    chunks = []
    t_start = time.perf_counter()
    first_audio_wall_time = None

    print(f"\n==> Start streaming text #{idx + 1}")
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        for i, (chunk, _, _) in enumerate(
            model.tts_model.generate_with_prompt_cache_streaming(
                target_text=text,
                prompt_cache=prompt_cache,
                cfg_value=2.0,
                inference_timesteps=inference_steps,
            )
        ):
            data = chunk.cpu().numpy().astype("float32")
            # chunk shape may be (1, T) or (T,) – squeeze then enforce mono
            data = np.squeeze(data)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim == 2:
                # if multiple channels exist, mix down to mono
                if data.shape[1] != 1:
                    data = data.mean(axis=1, keepdims=True)
            else:
                raise ValueError(f"Unexpected chunk shape: {data.shape}")

            stream.write(data)
            if first_audio_wall_time is None:
                first_audio_wall_time = time.perf_counter()

            chunks.append(data.squeeze(-1))
            print(f"  >> streamed chunk #{i + 1}, frames={len(data)}")

    t_end = time.perf_counter()
    metrics = {
        "time_to_first_audio": (first_audio_wall_time - t_start) if first_audio_wall_time else None,
        "total_wall_time": t_end - t_start,
    }

    if chunks:
        wav = np.concatenate(chunks, axis=0)
        out_path = f"output_streaming_cache_{idx + 1}.wav"
        sf.write(out_path, wav, SAMPLE_RATE)
        metrics["audio_duration"] = len(wav) / SAMPLE_RATE
        metrics["out_path"] = out_path

    return metrics


def main():
    USE_DENOISER = True  # 若 prompt 有噪音，開啟一次性去噪以提升音質
    inference_steps = 10
    # 1) Load model
    t0 = time.perf_counter()
    print("Loading VoxCPM model ...")
    model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")
    t_model_ready = time.perf_counter()
    print(f"Model ready in {(t_model_ready - t0):.3f}s")

    # 2) Prepare prompt (replace with your own files/text)
    prompt_wav_path = r"C:\Users\USER\VoxCPM\data\Lee4\Lee4_0.91.wav"
    prompt_text = """
    两者应该都对。但我后来我自己把这事情放在重点是你的性格，你的一切你的动力，那个是更根本的问题。
    而那个问题已经在这些事情发生之前，似乎已经注定了。你说这是悲观吗？这是一种命运论的，或者是人不要努力嘛，绝对不是我只是觉得你之所以为你自己，那就是一个天命。
    而这个天命来自于我们真的不知道有什么因素环境。天命应该每个人都有一个天命。那个人之所以会做这样子的准备或抓到一些机会来自于他自己的一些个性，一些性格。
    我觉得更根本的问题是你是谁？
    """

    # 3) (optional) denoise prompt once to提升品質
    if USE_DENOISER and getattr(model, "denoiser", None) is not None:
        print("Denoising prompt (one-time) ...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_f:
            clean_prompt_path = tmp_f.name
        model.denoiser.enhance(prompt_wav_path, output_path=clean_prompt_path)
        prompt_wav_path = clean_prompt_path
        print(f"Prompt denoised to temp file: {prompt_wav_path}")
    else:
        print("Skip denoise (either disabled or denoiser not available).")

    # 4) Build prompt cache once; reuse for every call
    print("Building prompt cache (one-time) ...")
    prompt_cache = model.tts_model.build_prompt_cache(
        prompt_text=prompt_text.strip(),
        prompt_wav_path=prompt_wav_path,
    )
    t_cache_ready = time.perf_counter()
    print(f"Prompt cache ready in {(t_cache_ready - t_model_ready):.3f}s")

    # 5) Texts to synthesize; append more as needed
    texts = [
        """您好，這邊是新光保全電話滿意度關懷。
        針對日前的服務，想耽誤您約一分鐘，確認幾個重點：
        第一，我方同仁是否有依約定時間到場？
        第二，整體服務 1到5分 您會給幾分滿意度？
        第三，若日後有新的居家安全或設備需求，您是否願意優先考慮新光保全？
        非常感謝您的協助，祝您一切順心，再見。""",
    ]

    # 6) Normalize text (與 streaming.py 一致)
    normalizer = TextNormalizer()
    texts = [normalizer.normalize(t.replace("\n", " ")) for t in texts]

    # 7) Stream each text reusing the same cache
    for idx, text in enumerate(texts):
        metrics = stream_text(model, prompt_cache, text, idx, inference_steps=inference_steps)
        print("  Metrics:")
        if metrics["time_to_first_audio"] is not None:
            print(f"    Time to first audio: {metrics['time_to_first_audio']:.3f}s")
        print(f"    Total wall time: {metrics['total_wall_time']:.3f}s")
        if "audio_duration" in metrics:
            print(f"    Audio duration: {metrics['audio_duration']:.3f}s")
        if "out_path" in metrics:
            print(f"    Saved wav: {metrics['out_path']}")


if __name__ == "__main__":
    main()
