"""
title: FLVM V1 [FAST LEARNING VIDEO MAKER]
author: stefanpietrusky
author_url: https://downchurch.studio/
version: 0.1
"""

import os
import re
import subprocess
import asyncio
import textwrap
import uuid
import json
import shutil
import hashlib
import numpy as np
import tempfile
import math
import edge_tts
import openai
import threading, time
import shlex
import random

from datetime import datetime
from flask import (Flask, request, jsonify, render_template_string, Response, send_from_directory)
from moviepy.editor import (VideoFileClip, ImageClip, CompositeVideoClip, ColorClip, TextClip, AudioFileClip, CompositeAudioClip,
    concatenate_videoclips, VideoClip)
from moviepy.audio.AudioClip import AudioClip 
from moviepy.video.fx import all as vfx
from moviepy.audio.fx.all import audio_loop, audio_fadein, audio_fadeout
from moviepy.video.compositing import transitions as transfx
from skimage.filters import sobel
from threading import Thread
from PIL import Image, ImageDraw, ImageFont
from proglog import ProgressBarLogger

class StatusLogger(ProgressBarLogger):
    def __init__(self, cb, total_frames=None):
        super().__init__()
        self._cb = cb
        self._total_frames = total_frames  

    def bars_callback(self, bar, attr, value, old_value=None):
        try:
            info = self.bars.get(bar, {})
            total = info.get('total') or self._total_frames or 0
            index = info.get('index') or 0
            if total:
                frac = max(0.0, min(1.0, index / total))
                self._cb("encode", frac)
        except Exception:
            pass

    def callback(self, **changes):
        p = changes.get('progress')
        if p is not None:
            try:
                self._cb("encode", float(p))
            except Exception:
                pass

status_lock = threading.Lock()

WEIGHTS = {
    "queued": 0.00,
    "tts": 0.25,
    "clips": 0.35,
    "transitions": 0.05,
    "encode": 0.30,
    "finalize": 0.05
}

def make_silent_mp3(path, duration=0.6):
    try:
        subprocess.run(
            [
                "ffmpeg","-hide_banner","-loglevel","error","-y",
                "-f","lavfi","-i","anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", str(duration),
                "-acodec","libmp3lame","-q:a","7",
                path
            ],
            check=True
        )
    except Exception as e:
        subprocess.run(
            [
                "ffmpeg","-hide_banner","-loglevel","error","-y",
                "-f","lavfi","-i","anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", str(duration),
                path.replace(".mp3",".wav")
            ],
            check=True
        )

def ffprobe_duration(path):
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True
        )
        return float(r.stdout.strip())
    except Exception:
        return None

def encode_with_progress(input_path, output_path, codec, framerate, progress_cb, audio_codec="aac"):
    total = ffprobe_duration(input_path)
    cmd = [
        "ffmpeg","-y","-i", input_path,
        "-c:v", codec,
        "-r", str(framerate),
        "-c:a", audio_codec,
        "-progress","pipe:1",
        "-loglevel","error",
        output_path
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True
    )
    last_emit = 0
    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_ms=") and total:
                micro = int(line.split("=")[1])
                cur = micro / 1_000_000.0
                frac = min(0.9999, cur / total) 
                now = time.time()
                if now - last_emit > 0.25:
                    progress_cb("encode", frac)
                    last_emit = now
            elif line.startswith("progress=") and line.endswith("end"):
                progress_cb("encode", 0.9999)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed (rc={proc.returncode})")
    progress_cb("encode", 1.0)

def _status_path(job_dir):
    return os.path.join(job_dir, "status.json")

def init_status(job_dir):
    data = {
        "state": "processing",
        "stage": "init",
        "percent": 0.0,
        "eta_seconds": None,
        "started": datetime.utcnow().isoformat()+"Z",
        "detail": {}
    }
    with status_lock:
        with open(_status_path(job_dir), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

_progress_history = {} 
_STAGE_HISTORY = {}
_RATE_SMOOTH = {} 

def update_status(job_dir, stage, sub_progress=None, done=False, error=None, extra=None):
    path = _status_path(job_dir)
    with status_lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"state":"processing","stage":stage}

        if error:
            data["state"] = "error"
            data["stage"] = stage
            data["error"] = str(error)
        elif done:
            data["state"] = "done"
            data["stage"] = stage
            data["percent"] = 100.0
            data["finished"] = datetime.utcnow().isoformat()+"Z"
            data.pop("eta_seconds", None)
        else:
            data["stage"] = stage
            percent = 0.0
            for key, w in WEIGHTS.items():
                if key == stage and sub_progress is not None:
                    percent += w * max(0.0, min(1.0, sub_progress))
                elif key == stage and sub_progress is None:
                    pass
                else:
                    order = list(WEIGHTS.keys())
                    if order.index(key) < order.index(stage):
                        percent += w
            data["percent"] = round(percent * 100, 2)

            history = _progress_history.setdefault(job_dir, [])
            now = time.time()
            sh = _STAGE_HISTORY.setdefault(job_dir, {}).setdefault(stage, [])
            if sub_progress is not None:
                sh.append((now, sub_progress))
                if len(sh) > 20:
                    sh[:] = sh[-20:]

                key = (job_dir, stage)
                rec = _RATE_SMOOTH.setdefault(key, {"rate": 0.0, "t": now, "p": sub_progress, "eta": None})

                dt = now - rec["t"]
                dp = sub_progress - rec["p"]

                if dt > 0 and dp >= 0:
                    inst_rate = dp / dt 
                    if rec["rate"] <= 0:
                        rec["rate"] = inst_rate
                    else:
                        rec["rate"] = 0.7 * rec["rate"] + 0.3 * inst_rate

                    rec["t"] = now
                    rec["p"] = sub_progress

                    if rec["rate"] > 1e-6:
                        new_eta = int(max(0.0, (1.0 - sub_progress) / rec["rate"]))
                        if rec["eta"] is not None:
                            low  = int(rec["eta"] * 0.75)
                            high = int(rec["eta"] * 1.25 + 1)
                            new_eta = max(low, min(high, new_eta))
                        rec["eta"] = new_eta
                        data["eta_seconds"] = rec["eta"]
                    else:
                        data.pop("eta_seconds", None)
                else:
                    data.pop("eta_seconds", None)
        if extra:
            data.setdefault("detail", {}).update(extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

os.environ["IMAGEMAGICK_BINARY"] = (
    r"C:\Program Files\..."
)

app = Flask(__name__)

def ensure_size(clip, video_size):
    return clip.resize(video_size)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def generate_script_with_ollama(topic, niveau, model_name="llama3.2"):
    system_msg = (
        "System: Denke zuerst intern Schritt für Schritt, gib aber nur das finale Skript aus."
        ' Antworte als JSON-Objekt mit "script" als Array von Sätzen.'
    )
    user_msg = json.dumps({"topic": topic, "niveau": niveau})
    prompt = f"{system_msg}\nUser: {user_msg}"

    proc = subprocess.Popen(
        ["ollama", "run", model_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout, stderr = proc.communicate(input=prompt, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"Ollama-Fehler: {stderr.strip()}")

    try:
        data = json.loads(stdout)
        script_list = data.get("script", [])
        return " ".join(script_list)
    except json.JSONDecodeError:
        raise RuntimeError(f"Ungültiges JSON von Ollama: {stdout}")


def generate_script_with_openai(topic, niveau, api_key, model_name="gpt-3.5-turbo"):
    openai.api_key = api_key
    system = {
        "role": "system",
        "content": (
            "Du sprichst Deutsch. Denke intern Schritt für Schritt, "
            'zeige nur das finale Skript als JSON-Objekt mit "script" als Array.'
        ),
    }
    user = {"role": "user", "content": json.dumps({"topic": topic, "niveau": niveau})}
    resp = openai.ChatCompletion.create(
        model=model_name, messages=[system, user], temperature=0.7, max_tokens=300
    )
    content = resp.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        script_list = data.get("script", [])
        return " ".join(script_list)
    except json.JSONDecodeError:
        raise RuntimeError(f"Ungültiges JSON von OpenAI: {content}")


def generate_script_with_gemini(topic, niveau, model="gemini-2.5-pro"):
    system = (
        "System: Intern Schritt für Schritt denken, aber nicht ausgeben. "
        'Gib nur ein JSON-Objekt mit "script" als Array von Sätzen zurück.'
    )
    payload = json.dumps({"topic": topic, "niveau": niveau})
    prompt = textwrap.dedent(
        f"""
    {system}
    User: {payload}
    """
    )

    exe = shutil.which("gemini") or shutil.which("gemini.cmd")
    if not exe:
        raise RuntimeError("Gemini-CLI nicht gefunden")

    result = subprocess.run(
        [exe, "-m", model],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    stdout = result.stdout.strip()

    cleaned = re.sub(r"^```(?:json)?", "", stdout).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        script_list = data.get("script", [])
        return " ".join(script_list)
    except json.JSONDecodeError:
        raise RuntimeError(f"Ungültiges JSON von Gemini: {cleaned}")


def split_script_into_sentences(script: str) -> list[str]:
    script = re.sub(
        r"(?is)^(hier (ist|kommt)|(nachfolgend|folgende(r)?))[^.?!]*[.?!]\s*",
        "",
        script,
    )
    sentences = re.split(r"(?<=[.!?])\s+", script.strip())
    filtered = [
        s.strip()
        for s in sentences
        if not re.match(r"(?i)^(hier (ist|kommt)|nachfolgend|folgende(r)?).*", s)
        and s.strip() != ""
    ]

    return filtered


def wrap_text_at_words(text, width):
    return "\n".join(
        textwrap.fill(
            text, width=width, break_long_words=False, replace_whitespace=False
        ).split("\n")
    )

async def convert_text_to_speech(text, output_file, voice, max_retries=5, base_delay=1.5, timeout=90):
    last_err = None
    for attempt in range(max_retries):
        try:
            comm = edge_tts.Communicate(text, voice=voice)
            await asyncio.wait_for(comm.save(output_file), timeout=timeout)
            return
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.75)
                await asyncio.sleep(delay)
            else:
                raise last_err

async def convert_multiple_texts_to_speech(sentences, audio_output_dir, voice):
    tasks = []
    for idx, sentence in enumerate(sentences):
        audio_file = os.path.join(audio_output_dir, f"sentence_{idx+1}.mp3")
        tasks.append(convert_text_to_speech(sentence, audio_file, voice))
    await asyncio.gather(*tasks)


def slide_in(clip, duration=1, video_size=(1280, 720)):
    vw, vh = video_size
    return clip.set_position(
        lambda t: (max(vw * (1 - t / duration), 0), "center")
    ).set_duration(clip.duration)

def cleanup_temp_files(directory):
    for root, _, files in os.walk(directory):
        for f in files:
            if "TEMP_MPY" in f:
                try:
                    os.remove(os.path.join(root, f))
                except OSError:
                    pass

def run_job(
    proj_dir,
    sentences,
    voice,
    video_size,
    framerate,
    codec,   
    transition,
    include_subs,
    font_factor,
    images,
    intro_path,
    intro_dur,
    outro_path,
    outro_dur,
    bg_music_path,
    output_video_file,
    ext,
    colors,
    audio_codec  
):
    try:
        async def tts_seq():
            errors = []
            for i, sentence in enumerate(sentences, start=1):
                update_status(
                    proj_dir, stage="tts",
                    sub_progress=(i - 1) / len(sentences),
                    extra={"audio_done": i - 1, "audio_total": len(sentences)}
                )
                out_path = os.path.join(proj_dir, "audio", f"sentence_{i}.mp3")
                try:
                    await convert_text_to_speech(sentence, out_path, voice)
                except Exception as e:
                    make_silent_mp3(out_path, duration=0.8)
                    errors.append(f"Sentence {i}: {type(e).__name__}: {e}")
                update_status(
                    proj_dir, stage="tts",
                    sub_progress=i / len(sentences),
                    extra={"audio_done": i, "audio_total": len(sentences)}
                )
            if errors:
                update_status(proj_dir, stage="tts", sub_progress=1.0, extra={"tts_warnings": errors})

        os.makedirs(os.path.join(proj_dir, "audio"), exist_ok=True)
        asyncio.run(tts_seq())
        audio_dir = os.path.join(proj_dir, "audio")
        audio_files = [os.path.join(audio_dir, f"sentence_{i+1}.mp3") for i in range(len(sentences))]

        for p in audio_files:
            if not os.path.isfile(p):
                make_silent_mp3(p, duration=0.8)

        def _progress_cb(stage, frac):
            update_status(proj_dir, stage=stage, sub_progress=frac)

        update_status(proj_dir, stage="queued", sub_progress=1.0)

        create_video_with_subtitles(
            sentences,
            audio_files,
            output_video_file,
            images,
            default_colors=colors,
            intro_path=intro_path,
            intro_duration=intro_dur,
            outro_path=outro_path,
            outro_duration=outro_dur,
            bg_music_path=bg_music_path,
            video_size=video_size,
            framerate=framerate,
            codec=codec,
            audio_codec=audio_codec,
            include_subtitles=include_subs,
            font_factor=font_factor,
            transition=transition,
            progress_cb=_progress_cb,
        )

        cleanup_temp_files(proj_dir)
        update_status(
            proj_dir, stage="finalize", sub_progress=1.0, done=True,
            extra={"video_url": f"/download/{os.path.basename(proj_dir)}/{os.path.basename(output_video_file)}"}
        )

    except Exception as e:
        update_status(proj_dir, stage="finalize", error=f"{type(e).__name__}: {e}")

def _find_ttf():
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\ARIAL.TTF",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None 

def ffmpeg_can_read_first_frame(path: str) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg","-v","error","-y","-ss","0","-i",path,"-frames:v","1","-f","null","-"],
            capture_output=True, text=True
        )
        return r.returncode == 0
    except Exception:
        return False

def sanitize_video_for_moviepy(src_path: str, dst_dir: str, target_size, fps=25) -> tuple[str, dict]:
    meta = {"sanitized": False, "fallback_still": False}
    if ffmpeg_can_read_first_frame(src_path):
        return src_path, meta

    base = os.path.splitext(os.path.basename(src_path))[0]
    remux = os.path.join(dst_dir, f"{base}_remux.mp4")
    try:
        subprocess.run(
            ["ffmpeg","-hide_banner","-loglevel","error","-y","-i",src_path,
             "-c","copy","-movflags","+faststart", remux],
            check=True
        )
        if ffmpeg_can_read_first_frame(remux):
            meta["sanitized"] = True
            return remux, meta
    except Exception:
        pass

    safe = os.path.join(dst_dir, f"{base}_safe.mp4")
    w, h = target_size
    try:
        subprocess.run(
            ["ffmpeg","-hide_banner","-loglevel","error","-y",
             "-i", src_path,
             "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}",
             "-c:v","libx264","-pix_fmt","yuv420p","-profile:v","high",
             "-c:a","aac","-movflags","+faststart", safe],
            check=True
        )
        if ffmpeg_can_read_first_frame(safe):
            meta["sanitized"] = True
            return safe, meta
    except Exception:
        pass

    still = os.path.join(dst_dir, f"{base}_firstframe.png")
    try:
        for ts in ("0", "0.5"):
            r = subprocess.run(
                ["ffmpeg","-hide_banner","-loglevel","error","-y","-ss", ts, "-i", src_path,
                 "-frames:v","1", still],
                capture_output=True, text=True
            )
            if r.returncode == 0 and os.path.isfile(still):
                meta["fallback_still"] = True
                return still, meta
    except Exception:
        pass

    return src_path, meta

def _measure_wrap(text, draw, font, max_width_px):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        tw, th = draw.textbbox((0,0), test, font=font)[2:]
        if tw <= max_width_px or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def render_text_image(
    text,
    max_width_px, 
    font_px=48,
    line_spacing=1.2,
    text_color=(255,255,255,255),
    bg_color=(0,0,0,160),   
    pad_x=24,
    pad_y=16,
    radius=16,
    stroke_width=2,
    stroke_fill=(0,0,0,255),
):
    ttf = _find_ttf()
    if ttf:
        font = ImageFont.truetype(ttf, font_px)
    else:
        try:
            font = ImageFont.truetype("arial.ttf", font_px)
        except Exception:
            font = ImageFont.load_default()

    tmp = Image.new("RGB", (max_width_px, font_px * 4), "black")
    d   = ImageDraw.Draw(tmp)

    lines = _measure_wrap(text, d, font, max_width_px)
    _, base_h = d.textbbox((0,0), "Ag", font=font)[2:]
    line_h = base_h
    total_text_h = int(line_h * (len(lines) + (len(lines)-1)*(line_spacing-1)))

    max_line_w = 0
    for ln in lines:
        tw, th = d.textbbox((0,0), ln, font=font)[2:]
        max_line_w = max(max_line_w, tw)

    W = max_line_w + 2*pad_x
    H = total_text_h + 2*pad_y

    img = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([0,0,W-1,H-1], radius=radius, fill=bg_color)

    y = pad_y
    for ln in lines:
        tw, th = draw.textbbox((0,0), ln, font=font)[2:]
        x = (W - tw) // 2
        if stroke_width > 0:
            draw.text((x, y), ln, font=font, fill=stroke_fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
        draw.text((x, y), ln, font=font, fill=text_color,     stroke_width=stroke_width, stroke_fill=stroke_fill)
        y += int(line_h * line_spacing)

    return img 

def hex_to_rgba(hex_color, alpha=255):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i+lv//3], 16) for i in range(0, lv, lv//3))
    return rgb + (alpha,)

def sanitize_video_for_moviepy(src_path, dst_dir, target_size, fps=25, max_try=2):
    try:
        probe = subprocess.run(
            ["ffmpeg","-v","error","-y","-ss","0","-i",src_path,"-frames:v","1","-f","null","-"],
            capture_output=True, text=True
        )
        if probe.returncode == 0:
            return src_path  

        base = os.path.splitext(os.path.basename(src_path))[0]
        safe_path = os.path.join(dst_dir, f"{base}_safe.mp4")
        w, h = target_size
        cmd = [
            "ffmpeg","-hide_banner","-loglevel","error","-y",
            "-i", src_path,
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}",
            "-c:v", "libx264","-pix_fmt","yuv420p","-profile:v","high",
            "-c:a", "aac","-movflags","+faststart",
            safe_path
        ]
        subprocess.run(cmd, check=True)
        probe2 = subprocess.run(
            ["ffmpeg","-v","error","-y","-ss","0","-i",safe_path,"-frames:v","1","-f","null","-"],
            capture_output=True, text=True
        )
        return safe_path if probe2.returncode == 0 else src_path
    except Exception:
        return src_path

def create_video_with_subtitles(
    sentences,
    audio_files,
    output_video_file,
    images,
    default_colors,
    intro_path=None,
    intro_duration=None,
    outro_path=None,
    outro_duration=None,
    bg_music_path=None,
    video_size=(1280, 720),
    framerate=15,
    codec="h264_nvenc",
    audio_codec="aac",
    include_subtitles=True,
    font_factor=0.05,
    transition="none",
    progress_cb=None
):

    def _fit(clip):
        return clip.resize(video_size).set_fps(framerate)

    def _silent_audio(dur):
        return AudioClip(lambda t: 0.0, duration=max(0.001, float(dur)), fps=44100)

    def _load_intro_or_outro_video(path, duration, framerate, fit, video_size):
        try:
            clip = (VideoFileClip(path).subclip(0, float(duration)) if duration else VideoFileClip(path))
            return fit(clip)
        except Exception:
            try:
                eps = max(1.0 / max(1, framerate * 4), 0.01)  
                clip = (VideoFileClip(path).subclip(eps, eps + float(duration)) if duration
                        else VideoFileClip(path).subclip(eps))
                return fit(clip)
            except Exception:
                try:
                    still = os.path.join(os.path.dirname(path),
                                        f"{os.path.splitext(os.path.basename(path))[0]}_firstframe.png")
                    for ts in ("0", "0.5"):
                        r = subprocess.run(
                            ["ffmpeg","-hide_banner","-loglevel","error","-y",
                            "-ss", ts, "-i", path, "-frames:v","1", still],
                            capture_output=True, text=True
                        )
                        if r.returncode == 0 and os.path.isfile(still):
                            return fit(ImageClip(still).set_duration(float(duration or 2.0)))
                except Exception:
                    pass
                return None

    clips = []
    per_clip_durations = []

    if intro_path:
        ext = os.path.splitext(intro_path)[1].lower()
        if ext in (".png",".jpg",".jpeg",".gif"):
            intro_clip = ImageClip(intro_path).set_duration(float(intro_duration or 0))
            intro_clip = _fit(intro_clip)
        else:
            intro_clip = _load_intro_or_outro_video(intro_path, intro_duration, framerate, _fit, video_size)
        if intro_clip:
            if intro_clip.audio is None:
                intro_clip = intro_clip.set_audio(_silent_audio(intro_clip.duration))
            clips.append(intro_clip)
            per_clip_durations.append(intro_clip.duration)

    if progress_cb:
        progress_cb("clips", 0.0)

    for idx, (sentence, audio_file) in enumerate(zip(sentences, audio_files)):
        hex_color = (default_colors[idx] if idx < len(default_colors) and default_colors[idx] else "#ffffff")
        rgb_color = hex_to_rgb(hex_color)

        audio_clip = AudioFileClip(audio_file)
        dur = max(0.2, float(audio_clip.duration))

        background = ColorClip(video_size, rgb_color).set_duration(dur)

        img_clip = None
        image_path = images[idx] if idx < len(images) else ""
        if image_path and os.path.isfile(image_path):
            ext = os.path.splitext(image_path)[1].lower()
            try:
                if ext == ".gif":
                    img_clip = VideoFileClip(image_path).fx(vfx.loop, duration=dur)
                else:
                    img_clip = ImageClip(image_path).set_duration(dur)
                    iw, ih = img_clip.size
                    vw, vh = video_size
                    scale = min(vw / iw, vh / ih)
                    img_clip = img_clip.resize(scale)
            except Exception as e:
                print(f"Fehler beim Laden des Bildes '{image_path}': {e}")
                img_clip = None

        if img_clip:
            img_clip = img_clip.set_position("center")
            video_bg = CompositeVideoClip([background, img_clip])
        else:
            video_bg = background

        if include_subtitles:
            font_px = max(36, int(video_size[1] * font_factor))
            max_w   = int(video_size[0] * 0.85)

            text_color = hex_to_rgba("#FFFFFF", 255)

            sub_img = render_text_image(
                wrap_text_at_words(sentence, width=5000),
                max_width_px=max_w,
                font_px=font_px,
                line_spacing=1.22,
                text_color=text_color,
                bg_color=(0, 0, 0, 160),
                pad_x=0,
                pad_y=0,
                radius=0,
                stroke_width=0,
                stroke_fill=(0, 0, 0, 0),
            )

            subtitle = ImageClip(np.array(sub_img)).set_duration(dur)

            safe_bottom = int(video_size[1] * 0.08)
            y_offset = video_size[1] - subtitle.h - safe_bottom

            subtitle_clip = subtitle.set_position(("center", max(0, y_offset)))

            video_clip = CompositeVideoClip([video_bg, subtitle_clip]).set_audio(audio_clip).set_duration(dur)
        else:
            video_clip = video_bg.set_audio(audio_clip).set_duration(dur)

        video_clip = slide_in(video_clip, duration=min(1.0, max(0.2, 0.2 * dur)), video_size=video_size)

        clips.append(video_clip)
        per_clip_durations.append(video_clip.duration)

        if progress_cb:
            progress_cb("clips", (idx + 1) / max(1, len(sentences)))

    if outro_path:
        ext = os.path.splitext(outro_path)[1].lower()
        if ext in (".png",".jpg",".jpeg",".gif"):
            outro_clip = ImageClip(outro_path).set_duration(float(outro_duration or 0))
            outro_clip = _fit(outro_clip)
        else:
            outro_clip = _load_intro_or_outro_video(outro_path, outro_duration, framerate, _fit, video_size)
        if outro_clip:
            if outro_clip.audio is None:
                outro_clip = outro_clip.set_audio(_silent_audio(outro_clip.duration))
            clips.append(outro_clip)
            per_clip_durations.append(outro_clip.duration)

    try:
        if transition == "crossfade":
            min_d = min(per_clip_durations) if per_clip_durations else 0
            fade_d = max(0.2, min(1.0, 0.3 * min_d)) if min_d > 0 else 0.3
            clips = [c.fx(vfx.fadein, fade_d).fx(vfx.fadeout, fade_d) for c in clips]
            final_video = concatenate_videoclips(clips, method="compose", padding=-fade_d)

        elif transition == "wipe":
            min_d = min(per_clip_durations) if per_clip_durations else 0
            dur = max(0.6, min(1.2, 0.25 * min_d)) if min_d > 0 else 0.6

            def wipe_between(a, b, duration):
                a_dur = float(a.duration or 0)
                b_dur = float(b.duration or 0)

                if a_dur > 0 and b_dur > 0:
                    duration = max(0.1, min(duration, a_dur, b_dur))
                else:
                    duration = 0.3 

                eps = max(1.0 / max(1, framerate * 4), 0.01) 

                try:
                    fa_last = a.get_frame(max(0, a_dur - 1e-3))
                except Exception:
                    try:
                        fa_last = a.get_frame(max(0, a_dur - 2 * eps))
                    except Exception:
                        fa_last = a.get_frame(max(0, a_dur * 0.99))

                try:
                    fb_first = b.get_frame(eps)
                except Exception:
                    try:
                        fb_first = b.get_frame(min(2 * eps, max(0, b_dur - 1e-3)))
                    except Exception:
                        fb_first = b.get_frame(max(0, min(b_dur * 0.01, b_dur - 1e-3)))

                def make_frame(t):
                    if duration <= 0:
                        return fa_last
                    w, h = video_size
                    x = int(w * (t / duration))
                    mask = np.zeros((h, w, 1), dtype="uint8")
                    mask[:, :max(0, min(w, x))] = 255
                    return (fa_last * (1 - mask / 255) + fb_first * (mask / 255)).astype("uint8")

                tclip = VideoClip(make_frame, duration=duration).set_fps(framerate).resize(video_size)
                tclip = tclip.set_audio(_silent_audio(duration))
                return tclip

            out = []
            if clips:
                out.append(clips[0])
                for nxt in clips[1:]:
                    out.append(wipe_between(out[-1], nxt, dur))
                    out.append(nxt)
                final_video = concatenate_videoclips(out, method="compose")
            else:
                final_video = ColorClip(video_size, (0, 0, 0)).set_duration(0.1)

            if progress_cb:
                progress_cb("transitions", 1.0)

        else:
            final_video = concatenate_videoclips(clips, method="compose")

        if progress_cb:
            progress_cb("transitions", 1.0)

    except Exception as e:
        raise RuntimeError(f"Transitions/Concatenate fehlgeschlagen: {e}")

    try:
        if bg_music_path and final_video.duration and final_video.duration > 0:
            intro_d = float(intro_duration or 0)
            outro_d = float(outro_duration or 0)
            total_dur = final_video.duration
            speech_start = max(0.0, intro_d)
            speech_end = max(speech_start, total_dur - max(0.0, outro_d))
            music_duration = max(0.0, speech_end - speech_start)

            if music_duration > 0.01:
                bg = AudioFileClip(bg_music_path)
                bg_looped = audio_loop(bg, duration=music_duration)
                last_dur = AudioFileClip(audio_files[-1]).duration if audio_files else 1.0
                bg_faded = audio_fadeout(audio_fadein(bg_looped, 0.8), min(2.0, last_dur))
                bg_final = bg_faded.set_start(speech_start)

                speech_audio = final_video.audio if final_video.audio is not None else _silent_audio(final_video.duration)
                combined_audio = CompositeAudioClip([speech_audio, bg_final])
                final_video = final_video.set_audio(combined_audio)
    except Exception as e:
        print(f"Warnung: Hintergrundmusik konnte nicht gemischt werden: {e}")

    if progress_cb:
        progress_cb("encode", 0.0)

    preset_kw = {}
    ffmpeg_params = []

    if codec == "libx264":
        preset_kw["preset"] = "ultrafast"
        ffmpeg_params += ["-pix_fmt", "yuv420p"]
    elif codec == "h264_nvenc":
        ffmpeg_params += [
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", "19",
            "-b:v", "20M",
            "-maxrate", "40M",
            "-bufsize", "40M",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p"
        ]

    total_frames = int((final_video.duration or 0) * framerate)
    logger = StatusLogger(progress_cb, total_frames=total_frames) if progress_cb else None

    temp_audio = os.path.join(os.path.dirname(output_video_file), "temp-audio.m4a")

    try:
        final_video.write_videofile(
            output_video_file,
            fps=framerate,
            codec=codec,
            audio_codec=audio_codec,
            threads=4,
            bitrate=None,
            logger=logger,
            ffmpeg_params=ffmpeg_params if ffmpeg_params else None,
            temp_audiofile=temp_audio,
            remove_temp=True,
            **preset_kw,
        )
    except Exception as e:
        raise RuntimeError(f"Encoding fehlgeschlagen: {e}")

    if progress_cb:
        progress_cb("encode", 1.0)

    return output_video_file

@app.route("/")
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"></script>
      <title>FLVM</title>
      <link rel="stylesheet" href="/styles.css">
    </head>
    <body>
      <div class="container">
        <h1>FAST LEARNING VIDEO MAKER V1</h1>

        <div class="video-config-row">
        <div class="video-config-item">
            <label for="backend-select">Backend:</label>
            <select id="backend-select">
            <option value="ollama">Lokal (Ollama)</option>
            <option value="openai">OpenAI API</option>
            <option value="gemini">Google Gemini CLI</option>
            </select>
        </div>
        <div class="video-config-item" id="ollama-model-div">
            <label for="ollama-model-select">Ollama-Modell:</label>
            <select id="ollama-model-select">
            <option>lade…</option>
            </select>
        </div>
        <div class="video-config-item" id="openai-key-div" style="display:none;">
            <label for="api-key-input">OpenAI-API-Key:</label>
            <input type="text" id="api-key-input" placeholder="sk-…" />
        </div>
        <div class="video-config-item" id="gemini-model-div" style="display:none;">
        <label for="gemini-model-select">Gemini-Modell:</label>
        <select id="gemini-model-select">
        <option value="gemini-2.5-pro" selected>gemini-2.5-pro</option>
        <option value="gemini-2.5-flash">gemini-2.5-flash</option>
        <option value="gemini-1.5-pro-002">gemini-1.5-pro-002</option>
        </select>
        </div>
        </div>

        <div class="video-config-row">
        <div class="video-config-item">
            <label for="resolution-select">Auflösung:</label>
            <select id="resolution-select">
            <option value="1280x720" selected>1280×720 (HD)</option>
            <option value="1920x1080">1920×1080 (Full-HD)</option>
            <option value="3840x2160">3840×2160 (4K)</option>
            <option value="720x1280">720×1280 (TikTok/Reels)</option>
            <option value="1080x1920">1080×1920 (YouTube Shorts)</option>
            </select>
        </div>

        <div class="video-config-item">
            <label for="codec-select">Video-Codec:</label>
            <select id="codec-select">
            <option value="h264_nvenc" selected>h264_nvenc (MP4)</option>
            <option value="libx264">libx264 (MP4/MOV)</option>
            <option value="mpeg4">mpeg4 (AVI)</option>
            <option value="libvpx-vp9">VP9 (WEBM)</option>
            <option value="prores_ks">ProRes (MOV)</option>
            </select>
        </div>

        <div class="video-config-item">
        <label for="voice-select">Stimme:</label>
        <select id="voice-select">
            <optgroup label="Deutsch">
            <option value="de-DE-ConradNeural">Conrad (m)</option>
            <option value="de-DE-KatjaNeural">Katja (w)</option>
            <option value="de-DE-SeraphinaMultilingualNeural">Seraphina (w)</option>
            </optgroup>
            <optgroup label="Englisch">
            <option value="en-US-AnaNeural">Ana (US, w)</option>
            <option value="en-GB-ThomasNeural">Thomas (UK, m)</option>
            <option value="en-US-BrianNeural">Brian (US, m)</option>
            </optgroup>
            <optgroup label="Italienisch">
            <option value="it-IT-ElsaNeural">Elsa (w)</option>
            <option value="it-IT-DiegoNeural">Diego (m)</option>
            <option value="it-IT-IsabellaNeural">Isabella (w)</option>
            </optgroup>
            <optgroup label="Spanisch">
            <option value="es-ES-ElviraNeural">Elvira (ES, w)</option>
            <option value="es-ES-AlvaroNeural">Álvaro (ES, m)</option>
            <option value="es-MX-JorgeNeural">Jorge (MX, m)</option>
            </optgroup>
            <optgroup label="Französisch">
            <option value="fr-FR-DeniseNeural">Denise (w)</option>
            <option value="fr-FR-HenriNeural">Henri (m)</option>
            <option value="fr-FR-RemyMultilingualNeural">Remy (m)</option>
            </optgroup>
        </select>
        </div>

        <div class="video-config-item">
        <label for="subtitle-select">Untertitel:</label>
        <select id="subtitle-select" name="subtitle">
            <option value="normal" selected>Ja (normal)</option>
            <option value="barrierefrei">Ja (barrierefrei)</option>
            <option value="none">Nein</option>
        </select>
        </div>

        <div class="video-config-item">
        <label for="transition-select">Übergangseffekt:</label>
        <select id="transition-select" name="transition">
            <option value="none" selected>Kein (harte Schnitte)</option>
            <option value="crossfade">Crossfade</option>
            <option value="wipe">Wipe</option>
        </select>
        </div>

        <div class="video-config-item">
            <label for="framerate-input">FPS:</label>
            <input type="text" id="framerate-input" placeholder="z.B. 15" class="duration-input"/>
        </div>

        </div>

        <label>Skript hochladen (optional .txt):</label>
        <div class="custom-file-container" id="script-file-container">
        <button type="button" class="file-btn">Durchsuchen</button>
        <span class="file-status">Keine Datei ausgewählt</span>
        <input type="file" id="script-file-input" accept=".txt" class="file-input">
        </div>

        <label>Hintergrundmusik (optional, .mp3/.wav):</label>
        <div class="custom-file-container" id="bg-music-container">
        <button type="button" class="file-btn">Durchsuchen</button>
        <span class="file-status" id="bg-music-status">Keine Datei ausgewählt</span>
        <input type="file" id="bg-music-input" class="file-input"
                accept=".mp3, .wav, audio/mpeg, audio/wav">
        </div>

        <label>Thema:</label>
        <input type="text" id="topic-input" placeholder="Thema eingeben">
        <label>Niveau:</label>
        <select id="niveau-select">
            <option value="einfach">Einfach</option>
            <option value="mittel" selected>Mittel</option>
            <option value="schwer">Schwer</option>
        </select>

        <button id="generateBtn" class="btn">Skript generieren</button>
        <div id="spinner" class="spinner" style="display:none;"></div>
        <div id="message-container" aria-live="polite" aria-atomic="true"></div>

        <div id="script-container" style="display:none;">
          <h2>Skript bearbeiten</h2>

        <div id="intro-config" class="config-section">
        <label>Intro (optional):</label>
        <div class="custom-file-container">
            <button type="button" class="file-btn">Durchsuchen</button>
            <span class="file-status" id="intro-file-status">Keine Datei ausgewählt</span>
            <input type="file"
                id="intro-file-input"
                class="file-input"
                accept="image/png,image/jpeg,image/gif,video/mp4,video/quicktime">

        <div id="intro-duration-container" class="duration-block" style="display: none; margin-top: 10px;">
        <input type="text"
                id="intro-duration-input"
                placeholder="z.B. 5"
                class="duration-input">
        <label for="intro-duration-input">Intro-Dauer (Sek.)</label>
        </div>
        </div>
        </div>

        <div id="sentences"></div>
        <button id="add-sentence-btn" type="button">Neuen Satz hinzufügen</button>

        <div id="outro-config" class="config-section" style="margin-top:20px;">
        <label>Outro (optional):</label>
        <div class="custom-file-container">
            <button type="button" class="file-btn">Durchsuchen</button>
            <span class="file-status" id="outro-file-status">Keine Datei ausgewählt</span>
            <input type="file"
                id="outro-file-input"
                class="file-input"
                accept="image/png,image/jpeg,image/gif,video/mp4,video/quicktime">

        <div id="outro-duration-container" class="duration-block" style="display: none; margin-top: 10px;">
        <input type="text"
                id="outro-duration-input"
                placeholder="z.B. 5"
                class="duration-input">
        <label for="outro-duration-input">Outro-Dauer (Sek.)</label>
        </div>
        </div>
        </div>

        <button id="generate-video-btn">Skript bestätigen und Video generieren</button>
        <div id="job-progress" class="job-progress hidden" aria-live="polite" aria-atomic="true">
        <div class="job-progress-row">
            <div id="job-progress-text" class="job-progress-text">Videogenerierung: 0% | Wartet | ETA: …</div>
            <div class="progress-outer"><div class="progress-inner" id="job-progress-bar"></div></div>
        </div>
        </div>
        </div>

        <div id="result-container" style="display:none;">
          <h2>Video erstellt</h2>
          <button id="download-video-btn">Download</button>
        </div>    
      </div>
      <script src="/script.js"></script>
    </body>
    </html>
    """
    return render_template_string(html_content)


@app.route("/styles.css")
def styles():
    css_content = """

    *, *::before, *::after {
        box-sizing: border-box;
    }

    body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f4;
        margin: 0;
        padding: 20px;
    }

    button {
        display: block;
        width: 100%;
        padding: 10px;
        margin: 5px 0;
        background-color: #ffffff;
        border: 3px solid #262626;
        color: #262626;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1em;
        transition: background-color 0.3s ease;
    }

    button:hover {
        background-color: #262626;
        border: 3px solid #262626;
        color: white;
    }

    .container {
        width: 90%;
        max-width: 800px;
        margin: auto;
        background: white;
        padding: 20px;
        border-radius: 8px;
        border: 3px solid #262626;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    .custom-file-container {
        width: 100%;
        padding: 10px;
        border: 3px dashed #262626;
        border-radius: 5px;
        text-align: center;
        background-color: #ffffff;
        margin: 5px 0;
        position: relative;
    }

    .custom-file-container.hover {
        border-color: #00B0F0; 
        background-color: #96DCF8; 
    }

    .custom-file-container .file-input {
        display: none;
    }

    .custom-file-container .file-btn {
        background-color: #ffffff;
        border: 3px solid #262626;
        width: auto; 
        display: inline-block;
        color: #262626;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .custom-file-container .file-btn:hover {
        background-color: #262626;
        border: 3px solid #262626;
        color: white;
    }

    .custom-file-container .file-btn.delete-sentence-btn {
        border-color: #f44336;
        color: #f44336;
        background-color: #ffffff;
    }
    .custom-file-container .file-btn.delete-sentence-btn:hover {
        background-color: #f44336;
        border-color: #f44336;
        color: #ffffff;
    }

    .color-picker-btn {
        display: inline-block;
        padding: 10px 20px;
        border: 3px solid #262626;
        border-radius: 5px;
        background: #ffffff;
        color: #262626;
        font-size: 1em;
        cursor: pointer;
        transition: background-color 0.3s, color 0.3s;
        width: auto; 
    }
    .color-picker-btn:hover {
        background-color: #262626;
        color: #ffffff;
    }

    .color-picker-info {
        margin-top: 1px; 
        font-size: 0.8em;
        line-height: 1; 
        color: #262626;
    }

    .sentence-container button.delete-sentence-btn {
        display: inline-block;
        padding: 10px 20px;
        margin: 5px 0;
        background-color: transparent;
        border: 3px solid #f44336;
        color: #f44336;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.9em;
        transition: background-color 0.2s ease;
    }

    .sentence-container button.delete-sentence-btn:hover {
        background-color: #f44336;
        color: #ffffff;
    }

    .custom-file-container .file-status {
        display: block;
        margin-top: 1px;
        font-size: 0.8em;
        color: #262626;
        text-align: center;
    }

    .custom-file-container .btn-input {
        display: none; 
    }

    .custom-file-container .btn-input {
        display: inline-block;
        width: auto;
        padding: 10px 20px;
        margin-left: 10px;
        border: 3px solid #262626;
        border-radius: 5px;
        font-size: 1em;
        background-color: #ffffff;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .custom-file-container .btn-input:hover {
        background-color: #262626;
        color: white;
    }

    .duration-block {
        display: flex;
        flex-direction: column;
        align-items: center; 
        margin: 10px auto;
    }

    .duration-block label {
        display: block;
        margin-top: 1px;
        font-size: 0.8em;
        color: #262626;
        text-align: center;
    }

    .duration-input {
        display: block;
        width: 60px !important; 
        padding: 10px 20px;
        border: 3px solid #262626;
        border-radius: 4px;
        box-sizing: border-box;
        align-items: center;
    }

    #download-video-btn {
        display: inline-block;
        width: auto; 
        padding: 10px 20px;
        background-color: #ffffff;
        border: 3px solid #262626;
        color: #262626;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1em;
        transition: background-color 0.3s ease;
        margin: 10px auto;
    }
    #download-video-btn:hover {
        background-color: #262626;
        border: 3px solid #262626;
        color: #ffffff;
    }

    h1, h2 {
        text-align: center;
        color: #262626;
    }

    .form-section label {
        display: block;
        margin: 10px 0 5px;
    }

    input[type="text"], 
    select, input[type="color"] {
        width: 100%;
        padding: 10px;
        margin-bottom: 10px;
        border: 3px solid #262626;
        border-radius: 5px;
        box-sizing: border-box;
        display: inline-block;
        font-family: Arial, sans-serif;
    }

    .spinner {
        border: 8px solid #262626;
        border-top: 8px solid #00B0F0;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }

    .sentence-container {
        cursor: move; 
        border: 3px solid #262626;
        border-radius: 5px;
        margin-bottom: 10px;
        background: #fff;
        overflow: hidden;
    }

    .sentence-container > summary {
        padding: 10px;
        cursor: pointer;
        font-weight: bold;
        list-style: none;
    }

    .sentence-container[open] > summary {
        background-color: #262626;
        color: #fff;
    }

    .sentence-body {
        padding: 10px;
        border-top: 1px solid #ccc;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    #sentences .sentence-container {
        margin-bottom: 10px;
        padding: 10px;
        border: 3px solid #262626;
        border-radius: 5px;
    }

    #sentences textarea {
        width: 100%;
        padding: 8px;
        margin: 5px 0;
        box-sizing: border-box;
        font-family: Arial, sans-serif;
        resize: none;
        border: 3px solid #262626;
        border-radius: 5px
    }

    #sentences .image-input {
        display: block;
        width: 100%;
        padding: 20px;
        margin: 5px 0;
        border: 3px dashed #262626;
        border-radius: 5px;
        text-align: center;
        cursor: pointer;
        background-color: #ffffff;
        transition: background-color 0.3s, border-color 0.3s;
    }

    #sentences .file-hint {
        display: block;
        margin-top: 1px;
        font-size: 0.9em;
        color: #262626;
    }

    #sentences .image-input.hover {
        border-color: #054b7a;
        background-color: #e0f7ff;
    }

    #sentences .image-input::file-selector-button {
        background-color: #2980b9;
        color: white;
        border: none;
        padding: 10px 20px;
        margin-right: 10px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    #sentences .image-input::file-selector-button:hover {
        background-color: #054b7a;
    }

    #result-container {
        text-align: center;
    }

    pre {
        background: #ffffff;
        padding: 10px;
        overflow-x: auto;
    }

    .video-config-row {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;  
        margin-bottom: 20px;   
        justify-content: space-between;
    }

    .video-config-item {
        flex: 1 1 0;       
        display: flex;
        flex-direction: column;
        min-width: 140px;  
    }

    .video-config-item > label {
        margin-bottom: 5px;
        color: #262626;
        font-weight: 500;
    }

    .video-config-row,
    .custom-file-container,
    .form-section label,
    #sentences + button,
    #script-container {
        margin-bottom: 10px;
    }

    #intro-config {
        margin-bottom: 15px;
    }

    #outro-config {
        margin-bottom: 15px;
    }

    input[type="text"],
    select,
    input[type="color"] {
        transition: border-color .2s ease, box-shadow .2s ease;
    }

    input[type="text"]:focus,
    select:focus,
    input[type="color"]:focus {
        outline: none; 
        border: 3px solid #00B0F0;  
        box-shadow: 0 0 0 3px rgba(0, 176, 240, .35);
    }

    #generateBtn {
        margin-top: 1rem; 
    }

    .error-message {
        display: block;
        color: #f44336;
        font-weight: bold;
        background-color: #ffffff; 
        padding: 10px; 
        border-radius: 5px; 
        border: 3px solid #f44336;
        text-align: center; 
        margin-top: 10px;
        margin-bottom: 10px;
        opacity: 1;
        transition: opacity 0.5s ease;
    }

    .error-message.hide {
        opacity: 0;
    }

    .hidden { display: none; }

    .job-progress {
        margin-top: 10px;
        background: #fff;
        border: 3px solid #262626;
        border-radius: 6px;
        padding: 12px;
    }

    .job-progress-text {
        margin-bottom: 8px;
        font-weight: 600;
        color: #262626;
    }

    .progress-outer {
        border: 3px solid #262626;  
        border-radius: 0;    
        background-color: #262626;  
        height: 20px;
        width: 100%;
        padding: 0;
    }

    .progress-inner {
        background-color: #00B0F0; 
        height: 100%;
        width: 0%;
        border-radius: 0;
        transition: width 0.3s ease;
    }

    """
    return Response(css_content, mimetype="text/css")


@app.route("/script.js")
def script():
    js_content = """

    const PROG_PREFIX = 'Videogenerierung: ';

    function getContrastYIQ(hex) {
    const c = hex.charAt(0) === '#' ? hex.substring(1) : hex;
    const r = parseInt(c.substr(0, 2), 16);
    const g = parseInt(c.substr(2, 2), 16);
    const b = parseInt(c.substr(4, 2), 16);
    const yiq = (r * 299 + g * 587 + b * 114) / 1000;
    return (yiq >= 128) ? '#262626' : '#ffffff';
    }

    function showError(msg, ms = 5000) {
    const container = document.getElementById('message-container');
    if (!container) return;

    const div = document.createElement('div');
    div.classList.add('error-message');
    div.textContent = msg;

    container.appendChild(div);

    setTimeout(() => {
        div.classList.add('hide');
        div.addEventListener('transitionend', () => {
        div.remove();
        });
    }, ms);
    }

    document.addEventListener('DOMContentLoaded', function() {

        function setupFileInputWithDuration(fileInputId, statusId, durationContainerId) {
            const input      = document.getElementById(fileInputId);
            const statusSpan = document.getElementById(statusId);
            const durBlock   = document.getElementById(durationContainerId);
            const container  = input.closest('.custom-file-container');
            const btn        = container.querySelector('.file-btn');
            const bgInput  = document.getElementById('bg-music-input');
            const bgBtn    = document.querySelector('#bg-music-container .file-btn');
            const bgStatus = document.getElementById('bg-music-status');

            bgBtn.addEventListener('click', () => {
            if (bgStatus.textContent !== 'Keine Datei ausgewählt') {
                bgInput.value       = '';
                bgStatus.textContent = 'Keine Datei ausgewählt';
                bgBtn.textContent    = 'Durchsuchen';
                bgBtn.classList.remove('delete-sentence-btn');
            } else {
                bgInput.click();
            }
            });

            bgInput.addEventListener('change', () => {
            if (!bgInput.files.length) return;
            const file = bgInput.files[0];
            bgStatus.textContent = file.name;
            bgBtn.textContent    = 'Löschen';
            bgBtn.classList.add('delete-sentence-btn');
            });

            btn.addEventListener('click', () => {
            if (btn.textContent === 'Löschen') {
                input.value = '';
                statusSpan.textContent = 'Keine Datei ausgewählt';
                btn.textContent = 'Durchsuchen';
                btn.classList.remove('delete-sentence-btn');
                durBlock.style.display = 'none';
            } else {
                input.click();
            }
            });

            if (input.type === 'file') {
            input.addEventListener('change', () => {
                if (!input.files.length) {
                btn.textContent = 'Durchsuchen';
                statusSpan.textContent = 'Keine Datei ausgewählt';
                durBlock.style.display = 'none';
                return;
                }
                const file = input.files[0];
                statusSpan.textContent = file.name;
                btn.textContent = 'Löschen';
                btn.classList.add('delete-sentence-btn');

                const ext = file.name.split('.').pop().toLowerCase();
                if (['png','jpg','jpeg','gif'].includes(ext)) {
                durBlock.style.display = 'block';
                } else {
                durBlock.style.display = 'none';
                }
            });
            }

            if (input.type === 'number') {
            input.style.display = 'none';
            input.min  = 1;
            input.step = 1;

            input.addEventListener('input', () => {
                const v = parseInt(input.value, 10);
                if (isNaN(v) || v < 1) {
                input.value = '';
                statusSpan.textContent = 'z.B. 5 Sek.';
                } else {
                statusSpan.textContent = v + ' Sek.';
                }
            });
            }
        }

        setupFileInputWithDuration(
            'intro-file-input',
            'intro-file-status',
            'intro-duration-container'
        );

        setupFileInputWithDuration(
            'outro-file-input',
            'outro-file-status',
            'outro-duration-container'
        );

        const backendSelect = document.getElementById('backend-select');
        const ollamaModelDiv = document.getElementById('ollama-model-div');
        const ollamaModelSelect = document.getElementById('ollama-model-select');
        const openaiKeyDiv = document.getElementById('openai-key-div');
        const apiKeyInput = document.getElementById('api-key-input');
        const generateScriptBtn = document.getElementById('generateBtn');
        const topicInput = document.getElementById('topic-input');
        const niveauSelect = document.getElementById('niveau-select');
        const spinner = document.getElementById('spinner');
        const scriptContainer = document.getElementById('script-container');
        const sentencesDiv = document.getElementById('sentences');
        const addSentenceBtn = document.getElementById('add-sentence-btn');
        const generateVideoBtn = document.getElementById('generate-video-btn');
        const resultContainer = document.getElementById('result-container');
        const downloadBtn = document.getElementById('download-video-btn');

        addSentenceBtn.addEventListener('click', () => {
        const count = sentencesDiv.children.length;
        const newContainer = createSentenceContainer("", count);
        sentencesDiv.appendChild(newContainer);
        updateSentenceLabels();
        });

        backendSelect.addEventListener('change', () => {
        const val = backendSelect.value;
        ollamaModelDiv.style.display   = val==='ollama' ? 'flex' : 'none';
        openaiKeyDiv.style.display     = val==='openai' ? 'flex' : 'none';
        document.getElementById('gemini-model-div').style.display =
            val==='gemini' ? 'flex' : 'none';
        });

        function clampedInput(selector) {
        const input = document.querySelector(selector);
        input.addEventListener('input', () => {
            let v = parseInt(input.value.replace(/\D/g, ''), 10);
            if (isNaN(v) || v < 0) v = 0;
            input.value = v;
        });
        }

        clampedInput('#intro-duration-input');
        clampedInput('#outro-duration-input');
        clampedInput('#framerate-input');

        fetch('/models')
        .then(res => res.json())
        .then(models => {
            ollamaModelSelect.innerHTML = models
            .map(m => `<option value="${m}">${m}</option>`)
            .join('');
        })
        .catch(() => {
            ollamaModelSelect.innerHTML = '<option value="">Fehler beim Laden</option>';
        });

        const scriptFileInput = document.getElementById('script-file-input');
        const scriptFileBtn   = document.querySelector('#script-file-container .file-btn');
        const scriptFileStatus = document.querySelector('#script-file-container .file-status');

        scriptFileBtn.addEventListener('click', () => {
        if (scriptFileStatus.textContent !== 'Keine Datei ausgewählt') {
            scriptFileInput.value = '';
            scriptFileStatus.textContent = 'Keine Datei ausgewählt';
            scriptFileBtn.textContent = 'Durchsuchen…'; 
            scriptFileBtn.classList.remove('delete-sentence-btn');
            scriptContainer.style.display = 'none'; 
            return;
        }
        scriptFileInput.click();
        });

        scriptFileInput.addEventListener('change', () => {
            const file = scriptFileInput.files[0];
            if (!file) return;
            if (!file.name.match(/\.txt$/i)) {
                showError('Nur .txt Dateien erlaubt.', 5000);
                scriptFileInput.value = '';
                return;
            }

            spinner.style.display = 'block';
            scriptFileStatus.textContent = file.name;
            scriptFileBtn.textContent = 'Löschen';
            scriptFileBtn.classList.add('delete-sentence-btn');
            
            const formData = new FormData();
            formData.append('script_file', file);
            
            fetch('/upload-script', {
                method: 'POST',
                body: formData
            })
                .then(res => res.json())
                .then(data => {
                spinner.style.display = 'none';
                if (data.error) {
                    showError(data.error, 5000);
                    return;
                }
                sentencesDiv.innerHTML = '';
                data.sentences.forEach((s, idx) => {
                    const container = createSentenceContainer(s.trim(), idx);
                    sentencesDiv.appendChild(container);
                });

                new Sortable(sentencesDiv, {
                    animation: 150,
                    handle: 'summary', 
                    onEnd: updateSentenceLabels
                });

                scriptContainer.style.display = 'block';
                })
                .catch(err => {
                spinner.style.display = 'none';
                console.error(err);
                showError('Fehler beim Hochladen des Skripts.', 5000);
            });
        });

        generateScriptBtn.addEventListener('click', function() {
            if (scriptFileInput.files.length) {
                return;
            }
            const topic = topicInput.value.trim();
            const niveau = niveauSelect.value;

            if (!topic) {
                showError("Bitte Thema eingeben.", 5000);
                spinner.style.display = 'none';
                return;
            }

            generateScriptBtn.insertAdjacentElement('afterend', spinner);
            spinner.style.display = 'block';
            scriptContainer.style.display = 'none';
            resultContainer.style.display = 'none';

            const payload = {
                backend: backendSelect.value,
                topic:   topicInput.value.trim(),
                niveau:  niveauSelect.value,
            };

            if (payload.backend === 'openai') {
                payload.api_key = apiKeyInput.value.trim();
            } else if (payload.backend === 'gemini') {
                payload.gemini_model = document.getElementById('gemini-model-select').value;
            } else {
                payload.ollama_model = ollamaModelSelect.value;
            }

            fetch('/generate-script', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(response => response.json())
            .then(data => {
                spinner.style.display = 'none';
                if (data.error) {
                    showError(data.error, 5000);
                    return;
                }

                sentencesDiv.innerHTML = "";

                data.sentences.forEach((text, idx) => {
                    const container = createSentenceContainer(text, idx);
                    sentencesDiv.appendChild(container);
                });

                addSentenceBtn.onclick = () => {
                    const count = sentencesDiv.children.length;
                    const newContainer = createSentenceContainer("", count);
                    sentencesDiv.appendChild(newContainer);
                    updateSentenceLabels();
                };

                scriptContainer.style.display = "block";
            })
            .catch(err => {
                spinner.style.display = 'none';
                console.error(err);
                showError("Fehler beim Generieren des Skripts.", 5000);
            });
        });

        generateVideoBtn.addEventListener('click', function() {
            if (generateVideoBtn.disabled) return;
            generateVideoBtn.disabled = true;
            resultContainer.style.display = 'none';
            downloadBtn.dataset.url = '';
            const formData   = new FormData();
            const topic      = topicInput.value.trim();
            const resolution = document.getElementById('resolution-select').value;
            const framerate  = document.getElementById('framerate-input').value;
            const codec      = document.getElementById('codec-select').value;
            const voice      = document.getElementById('voice-select').value;
            const subtitleOpt = document.getElementById('subtitle-select').value;
            const transition = document.getElementById('transition-select').value;
            const box  = document.getElementById('job-progress');
            const bar  = document.getElementById('job-progress-bar');
            const text = document.getElementById('job-progress-text');
            box.classList.remove('hidden');
            bar.style.width = '0%';
            text.textContent = `${PROG_PREFIX}0% | Wartet | ETA: …`;

            formData.append('topic', topic);
            formData.append('resolution', resolution);
            formData.append('framerate', framerate);
            formData.append('codec', codec);
            formData.append('voice', voice);
            formData.append('subtitle', subtitleOpt);
            formData.append('transition', transition);

            document.querySelectorAll('#sentences textarea')
            .forEach(el => formData.append('sentences[]', el.value.trim()));

            document.querySelectorAll('.bg-color-input')
            .forEach(input => formData.append('colors[]', input.value || '#ffffff'));

            document.querySelectorAll('#sentences .custom-file-container .file-input')
                .forEach((input, idx) => {
                if (input.files.length) {
                    formData.append(`image_${idx}`, input.files[0]);
                }
                });

            const introFile = document.getElementById('intro-file-input').files[0];
            const outroFile = document.getElementById('outro-file-input').files[0];
            const introDur  = document.getElementById('intro-duration-input').value;
            const outroDur  = document.getElementById('outro-duration-input').value;

            if (introFile) {
            formData.append('intro_file',     introFile);
            formData.append('intro_duration', introDur || 5);
            }
            if (outroFile) {
            formData.append('outro_file',     outroFile);
            formData.append('outro_duration', outroDur || 5);
            }

            const bgInput = document.getElementById('bg-music-input');
            if (bgInput.files.length) {
                formData.append('bg_music', bgInput.files[0]);
            }

            generateVideoBtn.insertAdjacentElement('afterend', spinner);
            spinner.style.display = 'block';

            fetch('/generate-video', { method:'POST', body: formData })
            .then(r=>r.json())
            .then(data=>{
                spinner.style.display='none';
                if(data.error){
                    showError(data.error, 6000);
                    generateVideoBtn.disabled = false;
                    return;
                }
                const jobId = data.job_id || (data.video_url ? data.video_url.split('/')[2] : null);
                if(!jobId){
                    showError("Keine job_id erhalten.", 6000);
                    generateVideoBtn.disabled = false;
                    return;
                }
                startPolling(jobId);
            })
            .catch(err=>{
                spinner.style.display='none';
                console.error(err);
                showError("Fehler beim Start des Jobs.", 6000);
                generateVideoBtn.disabled = false;
            });
        });

        downloadBtn.addEventListener('click', () => {
            const url = downloadBtn.dataset.url;
            if (url) window.location.href = url;
        });

        function updateSentenceLabels() {
        document.querySelectorAll('#sentences .sentence-container').forEach((container, idx) => {
            const summary = container.querySelector('summary');
            if (summary) summary.textContent = `Satz ${idx + 1}`;
            const ta = container.querySelector('textarea');
            if (ta) ta.dataset.index = idx;
        });
        }

        function enableDragDropOn(container) {
        const input = container.querySelector('input[type="file"]');
        container.addEventListener('dragover',   e => { e.preventDefault(); container.classList.add('hover'); });
        container.addEventListener('dragenter',  e => { e.preventDefault(); container.classList.add('hover'); });
        container.addEventListener('dragleave',  e =>    container.classList.remove('hover'));
        container.addEventListener('dragend',    e =>    container.classList.remove('hover'));
        container.addEventListener('drop',       e => {
            e.preventDefault();
            container.classList.remove('hover');
            const files = e.dataTransfer.files;
            if (!files.length) return;
            input.files = files;
            input.dispatchEvent(new Event('change'));
        });
        }

        function createSentenceContainer(text = "", index) {
        const details = document.createElement('details');
        details.className = "sentence-container";
        details.innerHTML = `
            <summary>Satz ${index + 1}</summary>
            <div class="sentence-body">
            <textarea data-index="${index}">${text}</textarea><br>
            <div class="color-picker-wrapper" style="text-align:center; margin:5px 0;">
                <button type="button" class="color-picker-btn">Seitenfarbe?</button>
                <input type="color" class="bg-color-input" value="#ffffff" style="display:none;">
                <div class="color-picker-info">(optional, Standard: Weiß)</div>
            </div>
            <label>Bild/GIF (optional):</label><br>
            <div class="custom-file-container">
                <button type="button" class="file-btn">Durchsuchen</button>
                <span class="file-status">Keine Datei ausgewählt</span>
                <input type="file" class="file-input" accept="image/png, image/jpeg, image/gif">
            </div><br>
            <button type="button" class="delete-sentence-btn">Satz löschen</button><br>
            </div>
        `;

        const colorBtn   = details.querySelector('.color-picker-btn');
        const colorInput = details.querySelector('.bg-color-input');
        colorBtn.addEventListener('click', () => colorInput.click());
        colorInput.addEventListener('input', () => {
            const hex = colorInput.value;
            colorBtn.style.backgroundColor = hex;
            colorBtn.style.color = getContrastYIQ(hex);
        });

        details.querySelector('.delete-sentence-btn')
            .addEventListener('click', () => {
            details.remove();
            updateSentenceLabels();
            });

        const fc = details.querySelector('.custom-file-container');
        const fileInput = fc.querySelector('.file-input');
        const fileStatus = fc.querySelector('.file-status');
        const fileBtn = fc.querySelector('.file-btn');
        fileBtn.addEventListener('click', () => {
            if (fileBtn.textContent.trim() === 'Löschen') {
            fileInput.value = "";
            fileStatus.textContent = "Keine Datei ausgewählt";
            fileBtn.textContent = "Durchsuchen";
            fileBtn.classList.remove('delete-sentence-btn');
            } else {
            fileInput.click();
            }
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            if (!file.type.match(/image\/(jpeg|png|gif)/)) {
                showError('Bitte nur .png, .jpg oder .gif Dateien hochladen.', 5000);
                fileInput.value = "";
                fileStatus.textContent = "Keine Datei ausgewählt";
                fileBtn.textContent = "Durchsuchen";
            } else {
                fileStatus.textContent = file.name;
                fileBtn.textContent = "Löschen";
                fileBtn.classList.add('delete-sentence-btn');
            }
            } else {
            fileStatus.textContent = "Keine Datei ausgewählt";
            fileBtn.textContent = "Durchsuchen";
            }
        });

        enableDragDropOn(fc);
        return details;
        }

        document.querySelectorAll('.custom-file-container')
                .forEach(enableDragDropOn);
    });

    function formatETA(sec){
    if(sec == null) return 'ETA: ...';
    const m = Math.floor(sec/60);
    const s = sec % 60;
    return `ETA: ${m>0? m+'m ': ''}${s}s`;
    }

    let pollTimer = null;

    function startPolling(jobId){
    const box  = document.getElementById('job-progress');
    const bar  = document.getElementById('job-progress-bar');
    const text = document.getElementById('job-progress-text');
    const genBtn = document.getElementById('generate-video-btn');

    box.classList.remove('hidden');

    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(()=>{
        fetch(`/status/${jobId}`)
        .then(r=>r.json())
        .then(st => {
        if (st.error) {
            showError(st.error, 6000);
            clearInterval(pollTimer);
            genBtn.disabled = false;
            box.classList.add('hidden');
            return;
        }

        const pct = Number(st.percent);
        const pctSafe = Number.isFinite(pct) ? pct : 0;
        const stageNames = {
            queued: 'Wartet',
            tts: 'Sprache',
            clips: 'Clips',
            transitions: 'Übergänge',
            encode: 'Kodierung',
            finalize: 'Finalisierung'
        };
        let stageTxt = stageNames[st.stage] || st.stage;
        if (st.state === 'done') stageTxt = 'Fertig';
        else if (st.state === 'error') stageTxt = 'Fehler';

        const etaStr = (st.eta_seconds != null) ? formatETA(st.eta_seconds) : 'ETA: …';

        bar.style.width = pctSafe.toFixed(2) + '%';
        text.textContent = `${PROG_PREFIX}${pctSafe.toFixed(1)}% | ${stageTxt} | ${etaStr}`;

        if (st.state === 'done') {
            clearInterval(pollTimer);
            box.classList.add('hidden');
            const dlBtn = document.getElementById('download-video-btn');
            if (st.detail && st.detail.video_url) {
            dlBtn.dataset.url = st.detail.video_url;
            } else if (st.video_url) {
            dlBtn.dataset.url = st.video_url;
            }
            document.getElementById('result-container').style.display = 'block';
            genBtn.disabled = false;
        }
        if (st.state === 'error') {
            clearInterval(pollTimer);
            genBtn.disabled = false;
            box.classList.add('hidden');
        }
        })
        .catch(e=> console.error(e));
    }, 1000);
    }
    """
    return Response(js_content, mimetype="application/javascript")

@app.route("/models", methods=["GET"])
def list_models():
    try:
        res = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        lines = res.stdout.splitlines()
        models = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            first_col = re.split(r"\s+", line)[0]
            if first_col.lower() in ("name", "tag", "model", "size"):
                continue
            parts = re.split(r"\s+", line)
            models.append(parts[0])
    except Exception as e:
        print("Fehler beim Abfragen der Modelle:", e)
        models = []
    return jsonify(models)

@app.route("/generate-script", methods=["POST"])
def generate_script():
    data = request.get_json()
    topic = data.get("topic", "").strip()
    niveau = data.get("niveau", "mittel").strip()
    backend = data.get("backend", "ollama")
    gemini_model = data.get("gemini_model", "gemini-pro")

    if not topic:
        return jsonify({"error": "Bitte Thema angeben."}), 400

    prompt = (
        f"Schreibe ein kurzes 60-Sekunden-Skript zum Thema '{topic}' "
        f"auf {niveau}em Niveau, ohne Zeitangaben, visuelle Hinweise, "
        "nicht zu lange Sätze, keine Meta-Antworten oder Formatierungen."
    )

    try:
        if backend == "openai":
            api_key = data.get("api_key", "").strip()
            if not api_key:
                return jsonify({"error": "API-Key für OpenAI fehlt."}), 400

            script = generate_script_with_openai(topic, niveau, api_key)

        elif backend == "gemini":
            script = generate_script_with_gemini(topic, niveau, model=gemini_model)

        else:
            script = generate_script_with_ollama(
                prompt, data.get("ollama_model", "llama3.2")
            )

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"{backend}-Fehler: {e.stderr}"}), 500
    except Exception as e:
        return jsonify({"error": f"{backend}-Fehler: {e}"}), 500

    sentences = split_script_into_sentences(script)
    return jsonify({"sentences": sentences})

@app.route("/upload-script", methods=["POST"])
def upload_script():
    if "script_file" not in request.files:
        return jsonify(error="Keine Datei im Request"), 400

    file = request.files["script_file"]
    if not file.filename.lower().endswith(".txt"):
        return jsonify(error="Nur .txt Dateien erlaubt"), 400

    text = file.read().decode("utf-8", errors="ignore")
    if not text.strip():
        return jsonify(error="Datei ist leer"), 400

    sentences = split_script_into_sentences(text)
    return jsonify(sentences=sentences)

@app.route("/generate-video", methods=["POST"])
def generate_video():
    def _test_ffmpeg_encode(vcodec: str, acodec: str, container_ext: str) -> bool:
        try:
            with tempfile.TemporaryDirectory() as td:
                out = os.path.join(td, f"probe.{container_ext}")
                fmt = {"mp4": "mp4", "webm": "webm", "mov": "mov"}.get(container_ext, container_ext)
                cmd = [
                    "ffmpeg","-hide_banner","-loglevel","error","-y",
                    "-f","lavfi","-i","color=size=320x240:rate=25:color=black",
                    "-f","lavfi","-i","anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-t","0.5",
                    "-c:v", vcodec,
                    "-c:a", acodec,
                    "-f", fmt, out
                ]
                r = subprocess.run(cmd, capture_output=True, text=True)
                return r.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 0
        except Exception:
            return False

    def _audio_for(ext: str) -> str:
        return {"webm": "libopus", "mov": "aac", "mp4": "aac"}.get(ext, "aac")

    def _pick_codec_fallback(requested_vcodec: str):
        default_ext = {
            "h264_nvenc": "mp4",
            "libx264": "mp4",
            "libvpx-vp9": "webm",
            "prores_ks": "mov",
        }.get(requested_vcodec, "mp4")

        candidates = [
            (requested_vcodec, default_ext),
            ("libx264", "mp4"),
            ("libvpx-vp9", "webm"),
            ("prores_ks", "mov"),
        ]

        seen = set()
        for vcodec, ext in candidates:
            if vcodec in seen:
                continue
            seen.add(vcodec)
            acodec = _audio_for(ext)
            if _test_ffmpeg_encode(vcodec, acodec, ext):
                return vcodec, acodec, ext

        return "libx264", "aac", "mp4"

    voice = request.form.get("voice", "de-DE-ConradNeural")
    topic = request.form.get("topic", "video").strip()
    sentences = request.form.getlist("sentences[]")
    resolution = request.form.get("resolution", "1280x720")
    width, height = map(int, resolution.split("x"))
    video_size = (width, height)
    framerate = int(request.form.get("framerate") or 15)
    requested_codec = request.form.get("codec", "h264_nvenc")
    transition = request.form.get("transition", "none")

    subtitle_opt = request.form.get("subtitle", "normal")
    if subtitle_opt == "none":
        include_subs = False
        font_factor = 0
    elif subtitle_opt == "barrierefrei":
        include_subs = True
        font_factor = 0.075
    else:
        include_subs = True
        font_factor = 0.05

    if not sentences:
        return jsonify({"error": "Keine Sätze übermittelt."}), 400

    chosen_vcodec, chosen_acodec, chosen_ext = _pick_codec_fallback(requested_codec)

    base_dir = "videos"
    os.makedirs(base_dir, exist_ok=True)

    safe_topic = re.sub(r"[^A-Za-z0-9_-]+", "_", topic) or "video"

    descriptor = json.dumps({
        "sentences": sentences,
        "voice": voice,
        "resolution": resolution,
        "requested_codec": requested_codec,
        "chosen_vcodec": chosen_vcodec,
        "chosen_acodec": chosen_acodec,
        "chosen_ext": chosen_ext,
        "transition": transition,
        "subtitle_opt": subtitle_opt,
        "colors": request.form.getlist("colors[]"),
        "num_images": sum(1 for k in request.files if k.startswith("image_") and request.files[k].filename),
    }, sort_keys=True).encode("utf-8")

    job_hash = hashlib.sha256(descriptor).hexdigest()[:16]
    proj_name = f"{safe_topic}_{job_hash}"
    proj_dir = os.path.join(base_dir, proj_name)
    uploads_dir = os.path.join(proj_dir, "uploads")
    audio_dir = os.path.join(proj_dir, "audio")
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    output_video_file = os.path.join(proj_dir, f"{proj_name}.{chosen_ext}")

    if os.path.isfile(output_video_file):
        return jsonify({
            "video_url": f"/download/{proj_name}/{proj_name}.{chosen_ext}",
            "job_id": proj_name,
            "status_url": f"/status/{proj_name}"
        }), 200

    init_status(proj_dir)
    update_status(proj_dir, stage="queued", sub_progress=0.0,
                  extra={"chosen_vcodec": chosen_vcodec, "chosen_acodec": chosen_acodec, "container": chosen_ext})

    bg_music_path = None
    if "bg_music" in request.files and request.files["bg_music"].filename:
        f = request.files["bg_music"]
        fn = f"{uuid.uuid4().hex}_{f.filename}"
        bg_music_path = os.path.join(uploads_dir, fn)
        f.save(bg_music_path)

    script_path = os.path.join(uploads_dir, "script.txt")
    if not os.path.isfile(script_path):
        with open(script_path, "w", encoding="utf-8") as sf:
            sf.write("\n".join(sentences))

    images = []
    for idx in range(len(sentences)):
        key = f"image_{idx}"
        if key in request.files and request.files[key].filename:
            img = request.files[key]
            fn = f"{uuid.uuid4().hex}_{img.filename}"
            path = os.path.join(uploads_dir, fn)
            img.save(path)
            images.append(path)
        else:
            images.append("")

    def _sec(val):
        try:
            return float(val)
        except:
            return 0.0

    intro_dur = _sec(request.form.get("intro_duration"))
    outro_dur = _sec(request.form.get("outro_duration"))

    def _normalize_sanitize(ret):
        return ret if (isinstance(ret, tuple) and len(ret) == 2) else (ret, {"sanitized": False, "fallback_still": False})

    intro_path = None
    if "intro_file" in request.files and request.files["intro_file"].filename:
        f = request.files["intro_file"]
        fn = f"{uuid.uuid4().hex}_{f.filename}"
        intro_path = os.path.join(uploads_dir, fn)
        f.save(intro_path)
        intro_path, intro_meta = _normalize_sanitize(
            sanitize_video_for_moviepy(intro_path, uploads_dir, video_size, framerate)
        )
        update_status(proj_dir, stage="queued", sub_progress=0.0, extra={"intro_sanitized": intro_meta})

    outro_path = None
    if "outro_file" in request.files and request.files["outro_file"].filename:
        f = request.files["outro_file"]
        fn = f"{uuid.uuid4().hex}_{f.filename}"
        outro_path = os.path.join(uploads_dir, fn)
        f.save(outro_path)
        outro_path, outro_meta = _normalize_sanitize(
            sanitize_video_for_moviepy(outro_path, uploads_dir, video_size, framerate)
        )
        update_status(proj_dir, stage="queued", sub_progress=0.0, extra={"outro_sanitized": outro_meta})

    colors = request.form.getlist("colors[]")

    worker_args = (
        proj_dir,
        sentences,
        voice,
        video_size,
        framerate,
        chosen_vcodec,
        transition,
        include_subs,
        font_factor,
        images,
        intro_path,
        intro_dur,
        outro_path,
        outro_dur,
        bg_music_path,
        output_video_file,
        chosen_ext,
        colors,
        chosen_acodec
    )
    Thread(target=run_job, args=worker_args, daemon=True).start()

    return jsonify({
        "job_id": proj_name,
        "status_url": f"/status/{proj_name}"
    }), 202

@app.route("/status/<proj>")
def job_status(proj):
    dirpath = os.path.join(os.getcwd(), "videos", proj)
    path = os.path.join(dirpath, "status.json")
    if not os.path.isfile(path):
        return jsonify({"error": "unknown job"}), 404
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)

@app.route("/download/<proj>/<filename>")
def download(proj, filename):
    dirpath = os.path.join(os.getcwd(), "videos", proj)
    return send_from_directory(dirpath, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

