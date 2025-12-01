#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Non-streaming TTS run for timing comparison with streaming.py.
Uses identical generation parameters but waits for the full waveform before playback.
"""

import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from voxcpm import VoxCPM

SAMPLE_RATE = 16000


def main():
    t_start = time.perf_counter()
    print("Loading VoxCPM model ...")
    model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")
    t_model_ready = time.perf_counter()

    text = '''您好，這邊是新光保全電話滿意度關懷。
    針對日前的服務，想耽誤您約一分鐘，確認幾個重點：
    第一，我方同仁是否有依約定時間到場？
    第二，整體服務 1到5分 您會給幾分滿意度？
    第三，若日後有新的居家安全或設備需求，您是否願意優先考慮新光保全？
    非常感謝您的協助，祝您一切順心，再見。'''

    prompt_wav_path = "C:\\Users\\USER\\VoxCPM\\data\\Lee4\\Lee4_0.91.wav"
    prompt_text = '''
    两者应该都对。但我后来我自己把这事情放在重点是你的性格，你的一切你的动力，那个是更根本的一问题。
    而那个问题已经在这些事情发生之前，似乎已经注定了。你说这是悲观吗？这是一种命运论的，或者是人不要努力嘛，绝对不是我只是觉得你之所以为你自己，那就是一个天命。
    而这个天命来自于我们真的不知道有什么因素环境。天命应该每个人都有一个天命。那个人之所以会做这样子的准备或抓到一些机会来自于他自己的一些个性，一些性格。
    我觉得更根本的问题是你是谁？'''

    print("Start non-streaming TTS (generate full audio first) ...")
    t_gen_start = time.perf_counter()
    wav = model.generate(
        text=text,
        cfg_value=2.0,
        inference_timesteps=10,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        normalize=True,
        denoise=True,
        retry_badcase=True,
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=6.0,
    )
    t_gen_end = time.perf_counter()
    wav = np.asarray(wav, dtype=np.float32)

    # Playback full audio after generation
    sd.play(wav, SAMPLE_RATE)
    t_first_audio = time.perf_counter()
    sd.wait()
    audio_duration = len(wav) / SAMPLE_RATE

    out_path = "output_non_streaming.wav"
    sf.write(out_path, wav, SAMPLE_RATE)
    print(f"Saved final wav to: {out_path}")
    t_total_end = time.perf_counter()

    print("VoxPCM TTS Non-streaming playback finished.")
    print(f'generate text: {text}')
    print(f"Time - model load: {(t_model_ready - t_start):.3f}s")
    print(f"Time - lag: {(t_first_audio - t_model_ready):.3f}s")
    print(f"Time - start -> first audio: {(t_first_audio - t_start):.3f}s")
    print(f"Audio duration (s): {audio_duration:.3f}")
    print(f"Time - total (including playback): {(t_total_end - t_start):.3f}s")


if __name__ == "__main__":
    main()
