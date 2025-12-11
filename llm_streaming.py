#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from voxcpm import VoxCPM
from voxcpm.utils.llm_bridge import stream_llm_sentences

SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM -> VoxCPM streaming demo")
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="使用者輸入的提示；若加上 --use-asr 則會忽略此參數",
    )
    parser.add_argument(
        "--llm-id",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Hugging Face 上的超小 LLM id",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="LLM 最大生成 token 數",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=80,
        help="若未遇到句號時的最大片段長度",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="你是個簡潔友善的語音助手，請用中文直接回答問題，不要翻譯成英文，也不要額外解釋。",
        help="系統提示，預設強制中文回答",
    )
    parser.add_argument("--cfg-value", type=float, default=2.0, help="VoxCPM cfg 引導值")
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="VoxCPM 推理步數",
    )
    parser.add_argument("--prompt-wav-path", type=str, default=None, help="聲線提示音檔")
    parser.add_argument("--prompt-text", type=str, default=None, help="聲線提示文字")
    parser.add_argument(
        "--output",
        type=str,
        default="output_llm_streaming.wav",
        help="輸出 wav 路徑",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="停用提示音去噪",
    )

    # === 新增：ASR 相關參數 ===
    parser.add_argument(
        "--use-asr",
        action="store_true",
        help="使用 FunASR paraformer-zh 從麥克風錄音取得 prompt（會忽略 --prompt）",
    )
    parser.add_argument(
        "--asr-duration",
        type=float,
        default=5.0,
        help="麥克風錄音秒數（搭配 --use-asr 使用）",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="cuda:0",
        help="FunASR paraformer-zh 使用的 device（例如 cuda:0 或 cpu）",
    )

    return parser.parse_args()


def load_llm(llm_id: str):
    print(f"Loading LLM: {llm_id}")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        import accelerate  # noqa: F401

        device_map = "auto" if torch.cuda.is_available() else None
    except ImportError:
        device_map = None
        print("accelerate 未安裝，改用 CPU 設置；若要自動分配 GPU，請 `pip install accelerate`")
    model = AutoModelForCausalLM.from_pretrained(
        llm_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    return model, tokenizer


# === 新增：載入 FunASR paraformer-zh（用 GPU） ===
def load_asr(asr_device: str = "cuda:0"):
    print(f"Loading FunASR paraformer-zh on {asr_device} ...")
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device=asr_device,
    )
    return model


# === 新增：從麥克風錄音 ===
def record_from_mic(path: str, duration: float = 5.0, sample_rate: int = SAMPLE_RATE):
    print(f"開始錄音 ({duration:.1f} 秒)...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    sf.write(path, audio, sample_rate)
    print(f"錄音結束，已儲存到 {path}")

https://github.com/Leyan47/vocpcm/tree/main
# === 新增：用 FunASR paraformer-zh 做語音轉文字 ===
def asr_transcribe(asr_model, wav_path: str) -> str:
    res = asr_model.generate(input=wav_path, batch_size_s=300)
    if not res:
        return ""
    text = res[0].get("text", "")
    if text:
        text = rich_transcription_postprocess(text)
    return text.strip()


def main():
    args = parse_args()

    # === 1) 先決定 user_prompt：可以用 --prompt 或用 ASR 從麥克風來 ===
    if args.use_asr:
        asr_model = load_asr(args.asr_device)
        asr_wav_path = "asr_input.wav"
        record_from_mic(asr_wav_path, duration=args.asr_duration)
        user_prompt = asr_transcribe(asr_model, asr_wav_path)
        print(f"[ASR] 識別文字：{user_prompt}")
        if not user_prompt:
            print("ASR 沒有識別出文字，結束程式。")
            return
        del asr_model
        torch.cuda.empty_cache()
    else:
        if not args.prompt:
            raise ValueError("--prompt 必填，或加上 --use-asr 從麥克風輸入")
        user_prompt = args.prompt

    # === 2) 準備丟進 LLM 的 messages ===
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # === 3) 以下維持原本的 LLM + VoxCPM streaming 流程 ===
    t0 = time.perf_counter()
    model, tokenizer = load_llm(args.llm_id)
    llm_ready = time.perf_counter()

    print("Loading VoxCPM TTS ...")
    tts = VoxCPM.from_pretrained(load_denoiser=not args.no_denoise)
    tts_ready = time.perf_counter()

    chunks = []
    first_audio_wall_time = None
    print("開始串流：LLM 生成 -> VoxCPM 播放")

    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        for i, sentence in enumerate(
            stream_llm_sentences(
                model,
                tokenizer,
                messages,
                max_new_tokens=args.max_new_tokens,
                chunk_chars=args.chunk_chars,
            )
        ):
            print(f"  [LLM] 片段 {i+1}: {sentence}")
            for j, chunk in enumerate(
                tts.generate_streaming(
                    text=sentence,
                    cfg_value=args.cfg_value,
                    inference_timesteps=args.inference_timesteps,
                    prompt_wav_path=args.prompt_wav_path,
                    prompt_text=args.prompt_text,
                    normalize=True,
                    denoise=not args.no_denoise,
                    retry_badcase=True,
                    retry_badcase_max_times=3,
                    retry_badcase_ratio_threshold=6.0,
                )
            ):
                data = np.asarray(chunk, dtype="float32")
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                stream.write(data)
                if first_audio_wall_time is None:
                    first_audio_wall_time = time.perf_counter()
                chunks.append(data.squeeze(-1))
                print(f"    [TTS] streamed chunk {j+1}, frames={len(data)}")

    if chunks:
        wav = np.concatenate(chunks, axis=0)
        sf.write(args.output, wav, SAMPLE_RATE)
        audio_duration = len(wav) / SAMPLE_RATE
        print(f"Saved final wav: {args.output}, duration={audio_duration:.2f}s")
    else:
        print("No audio chunks generated.")

    t_end = time.perf_counter()
    print(f"Time - load LLM: {llm_ready - t0:.3f}s")
    print(f"Time - load TTS: {tts_ready - llm_ready:.3f}s")
    if first_audio_wall_time is not None:
        print(f"Time - lag (LLM ready -> first audio): {first_audio_wall_time - llm_ready:.3f}s")
        print(f"Time - start -> first audio: {first_audio_wall_time - t0:.3f}s")
    print(f"Time - total: {t_end - t0:.3f}s")


if __name__ == "__main__":
    main()
# e.g.
# python llm_streaming.py --prompt "介紹一下今天的天氣" --prompt-wav-path data/Lee4/Lee4_0.91.wav --prompt-text "两者应该都对。但我后来我自己把这事情放在重点是你的性格，你的一切你的动力，那个是更根本的问题。而那个问题已经在这些事情发生之前，似乎已经注定了。你说这是悲观吗？这是一种命运论的，或者是人不要努力嘛，绝对不是我只是觉得你之所以为你自己，那就是一个天命。而这个天命来自于我们真的不知道有什么因素环境。天命应该每个人都有一个天命。那个人之所以会做这样子的准备或抓到一些机会来自于他自己的一些个性，一些性格。我觉得更根本的问题是你是谁？"
# python llm_streaming.py --use-asr --asr-duration 5 --prompt-wav-path data/Lee4/Lee4_0.91.wav --prompt-text "两者应该都对。但我后来我自己把这事情放在重点是你的性格，你的一切你的动力，那个是更根本的问题。而那个问题已经在这些事情发生之前，似乎已经注定了。你说这是悲观吗？这是一种命运论的，或者是人不要努力嘛，绝对不是我只是觉得你之所以为你自己，那就是一个天命。而这个天命来自于我们真的不知道有什么因素环境。天命应该每个人都有一个天命。那个人之所以会做这样子的准备或抓到一些机会来自于他自己的一些个性，一些性格。我觉得更根本的问题是你是谁？"
