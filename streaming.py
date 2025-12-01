#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
import sounddevice as sd
import time
from voxcpm import VoxCPM

SAMPLE_RATE = 16000

def main():
    # 1) 載入模型（跟你原本的一樣）
    t_start = time.perf_counter()
    print("Loading VoxCPM model ...")
    model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")
    t_model_ready = time.perf_counter()

    # 你要念的文字
    text = '''您好，這邊是新光保全電話滿意度關懷。
    針對日前的服務，想耽誤您約一分鐘，確認幾個重點：
    第一，我方同仁是否有依約定時間到場？
    第二，整體服務 1到5分 您會給幾分滿意度？
    第三，若日後有新的居家安全或設備需求，您是否願意優先考慮新光保全？
    非常感謝您的協助，祝您一切順心，再見。'''

    # 如果要 voice cloning，就把這兩個改成你自己的檔案＆文本
    prompt_wav_path = "C:\\Users\\USER\\VoxCPM\\data\\Lee4\\Lee4_0.91.wav"
    prompt_text = '''
    两者应该都对。但我后来我自己把这事情放在重点是你的性格，你的一切你的动力，那个是更根本的一问题。
    而那个问题已经在这些事情发生之前，似乎已经注定了。你说这是悲观吗？这是一种命运论的，或者是人不要努力嘛，绝对不是我只是觉得你之所以为你自己，那就是一个天命。
    而这个天命来自于我们真的不知道有什么因素环境。天命应该每个人都有一个天命。那个人之所以会做这样子的准备或抓到一些机会来自于他自己的一些个性，一些性格。
    我觉得更根本的问题是你是谁？'''

    # 2) 開一個輸出串流，邊生邊寫出去
    chunks = []  # 同時保留 chunks，方便最後存檔
    print("Start streaming TTS and playing to speaker ...")
    t_gen_start = time.perf_counter()
    first_audio_wall_time = None

    # channels=1 因為 VoxCPM 是單聲道；dtype 用 float32
    with sd.OutputStream(samplerate=SAMPLE_RATE,
                         channels=1,
                         dtype="float32") as stream:
        for i, chunk in enumerate(
            model.generate_streaming(
                text=text,
                cfg_value=2.0,
                inference_timesteps=30,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
                normalize=True,
                denoise=True,
                retry_badcase=True,
                retry_badcase_max_times=3,
                retry_badcase_ratio_threshold=6.0,
            )
        ):
            # chunk 通常是 1D NumPy array，把它整理成 (frames, 1) 給 sounddevice
            data = np.asarray(chunk, dtype="float32")
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            # 3) 直接寫到輸出串流 -> 喇叭就會發出目前這一小段
            stream.write(data)
            if first_audio_wall_time is None:
                first_audio_wall_time = time.perf_counter()

            # 4) 同時保留一份，等等要拼回完整 wav
            chunks.append(data.squeeze(-1))

            # debug 用
            print(f"  >> streamed chunk #{i+1}, frames={len(data)}")

    print("VoxPCM TTS Streaming playback finished.")
    print(f'generate text: {text}')
    # t_gen_end = time.perf_counter()
    # if first_audio_wall_time is not None:
    #     elapsed = first_audio_wall_time - t_start

    # 5) 把所有 chunk 拼起來，順便寫一份 wav 到磁碟
    audio_duration = None
    if chunks:
        wav = np.concatenate(chunks, axis=0)
        out_path = "output_streaming_realtime.wav"
        sf.write(out_path, wav, SAMPLE_RATE)
        audio_duration = len(wav) / SAMPLE_RATE
        # print(f"Saved final wav to: {out_path}")
    else:
        print("No chunks generated, nothing saved.")

    print(f"Time - model load: {(t_model_ready - t_start):.3f}s")
    if first_audio_wall_time is not None:
        print(f"Time - lag: {(first_audio_wall_time - t_model_ready):.3f}s")
        print(f"Time - start->first audio: {(first_audio_wall_time - t_start):.3f}s")
    else:
        print("Time - start -> first audio: N/A (no audio played)")
        print("Time - lag: N/A (no audio played)")
    if audio_duration is not None:
        print(f"Audio duration (s): {audio_duration:.3f}")
    t_total_end = time.perf_counter()
    print(f"Time - total (including playback): {(t_total_end - t_start):.3f}s")

if __name__ == "__main__":
    main()
