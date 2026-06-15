import argparse
import re
import sys
import time
import random
import os
import wave
import threading
import subprocess
import tempfile
import json
import io
import tkinter as tk
from flask import Flask, request, jsonify
from flask_cors import CORS
from tkinter import ttk

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# OpenAI 제거 — Gemini로 통합
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()


# =========================
# LLM 설정 (OpenAI — 경고 문장 생성용)
# =========================
api_key_h = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gemini-3.1-flash-lite"

# =========================
# Gemini TTS 설정 (음성 발화용)
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"

# Gemini 클라이언트 (키 있을 때만 초기화)
_gemini_client = None
def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None and GEMINI_API_KEY:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


# =========================
# 경고 옵션 설정 (전역)
# 피그마 연동 시 이 값들을 외부에서 주입하면 됩니다
# =========================
class WarningOptions:
    """
    말투   : "부드러운" / "중간" / "단호"
    길이   : "단문"    / "중문" / "장문"
    voice  : Gemini TTS voice 이름 (Kore / Puck / Charon / Aoede / Achird / Sulafat)
    볼륨   : 정수 0~100
    """
    TONE_OPTIONS   = ["부드러운", "중간", "단호"]
    LENGTH_OPTIONS = ["단문", "중문", "장문"]

    # Gemini TTS 보이스 옵션 (성격별 6개)
    VOICE_OPTIONS = {
        "Kore":    "단호한",
        "Puck":    "경쾌한",
        "Charon":  "정보전달형",
        "Aoede":   "산뜻한",
        "Achird":  "친근한",
        "Sulafat": "따뜻한",
    }
    VOICE_FALLBACK = "Kore"

    def __init__(self):
        self.tone       = "중간"
        self.length     = "중문"
        self.voice      = "Kore"   # Gemini TTS voice name
        self.volume     = 80       # 볼륨 0~100
        self.escalation = True     # 단계적 경고 강화
        self.positive   = True     # 칭찬 피드백
        self.delay      = 3        # 칭찬 발화 딜레이(초)
        self.threshold  = 2        # 감지 임계값(초)
        self.rate       = 180      # 말하기 속도
        # 사용자 프로파일
        self.age        = "30대"   # 연령: 20대/30대/40대/50대+
        self.driveStyle = "Normal" # 운전 스타일: Active/Normal/Comfort
        self.experience = "일반"   # 운전 경험: 입문/일반/숙련

    def get_voice(self) -> str:
        return self.voice if self.voice in self.VOICE_OPTIONS else self.VOICE_FALLBACK

    def summary(self) -> str:
        return (f"[옵션] 말투={self.tone} | 길이={self.length} | "
                f"보이스={self.voice}({self.VOICE_OPTIONS.get(self.voice,'?')}) | "
                f"볼륨={self.volume} | "
                f"프로파일={self.age}/{self.driveStyle}/{self.experience}")


# 전역 옵션 인스턴스
OPTIONS = WarningOptions()


# =========================
# Flask 설정 서버 (HTML UI 연동)
# =========================
app = Flask(__name__)
CORS(app)  # HTML에서 fetch() 허용

@app.route('/get-options', methods=['GET'])
def get_options():
    """현재 OPTIONS 값을 HTML로 전달"""
    return jsonify({
        'tone':       OPTIONS.tone,
        'length':     OPTIONS.length,
        'voice':      OPTIONS.voice,
        'volume':     OPTIONS.volume,
        'escalation': OPTIONS.escalation,
        'positive':   OPTIONS.positive,
        'delay':      OPTIONS.delay,
        'threshold':  OPTIONS.threshold,
        'rate':       OPTIONS.rate,
        'age':        OPTIONS.age,
        'driveStyle': OPTIONS.driveStyle,
        'experience': OPTIONS.experience,
    })

@app.route('/set-options', methods=['POST'])
def set_options():
    """HTML 저장 버튼 → OPTIONS 업데이트"""
    data = request.get_json(force=True)
    if not data:
        return jsonify({'status': 'error', 'msg': 'no data'}), 400

    if 'tone'       in data: OPTIONS.tone       = data['tone']
    if 'length'     in data: OPTIONS.length      = data['length']
    if 'voice'      in data: OPTIONS.voice       = data['voice']
    if 'volume'     in data: OPTIONS.volume      = int(data['volume'])
    if 'escalation' in data: OPTIONS.escalation  = bool(data['escalation'])
    if 'positive'   in data: OPTIONS.positive    = bool(data['positive'])
    if 'delay'      in data: OPTIONS.delay       = int(data['delay'])
    if 'threshold'  in data: OPTIONS.threshold   = float(data['threshold'])
    if 'rate'       in data: OPTIONS.rate        = int(data['rate'])
    if 'age'        in data: OPTIONS.age         = data['age']
    if 'driveStyle' in data: OPTIONS.driveStyle  = data['driveStyle']
    if 'experience' in data: OPTIONS.experience  = data['experience']

    print(f"\n⚙️  [HTML → Python] 설정 업데이트: {OPTIONS.summary()}")
    return jsonify({'status': 'ok', 'options': {
        'tone': OPTIONS.tone, 'length': OPTIONS.length,
        'voice': OPTIONS.voice, 'volume': OPTIONS.volume,
    }})


import queue
alert_queue = queue.Queue()  # 경고 이벤트 전달용


@app.route('/preview-voice', methods=['POST'])
def preview_voice():
    """보이스 카드 클릭 시 호출 — 경고음 없이 샘플 문장만 재생"""
    data   = request.get_json(silent=True) or {}
    voice  = data.get('voice') or OPTIONS.get_voice()
    volume = int(data.get('volume', OPTIONS.volume))

    sample = "안녕하세요. 현재 선택된 보이스로 메시지가 전달됩니다."

    def _play():
        ok = _speak_gemini(sample, voice, volume)
        if not ok:
            _speak_fallback(sample)

    threading.Thread(target=_play, daemon=True).start()
    return jsonify({"status": "ok", "voice": voice})


@app.route('/alert-stream')
def alert_stream():
    """SSE 엔드포인트 — HTML이 여기 연결해서 실시간 경고 수신"""
    def event_generator():
        while True:
            try:
                data = alert_queue.get(timeout=30)
                yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"  # 연결 유지
    from flask import Response, stream_with_context
    return Response(stream_with_context(event_generator()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})

def run_flask():
    """백그라운드 스레드에서 Flask 실행 (포트 5050)"""
    app.run(host='127.0.0.1', port=5050, debug=False, use_reloader=False)



# =========================
# 옵션 설정 UI (tkinter)
# =========================
def open_options_window():
    """옵션 설정 팝업 창. 메인 루프 시작 전 또는 별도 스레드에서 호출."""

    win = tk.Tk()
    win.title("경고 옵션 설정")
    win.resizable(False, False)
    win.geometry("400x440")
    win.configure(bg="#1e1e2e")

    LABEL_STYLE = {"bg": "#1e1e2e", "fg": "#cdd6f4", "font": ("Helvetica", 12, "bold")}
    FRAME_STYLE = {"bg": "#1e1e2e"}

    def section(parent, title):
        tk.Label(parent, text=title, **LABEL_STYLE).pack(anchor="w", padx=20, pady=(14, 2))

    # ── 말투 ──────────────────────────────────────────────────────
    section(win, "🗣️  말투")
    tone_var = tk.StringVar(value=OPTIONS.tone)
    tone_frame = tk.Frame(win, **FRAME_STYLE)
    tone_frame.pack(anchor="w", padx=20)
    for opt in WarningOptions.TONE_OPTIONS:
        tk.Radiobutton(
            tone_frame, text=opt, variable=tone_var, value=opt,
            bg="#1e1e2e", fg="#89dceb", selectcolor="#313244",
            activebackground="#1e1e2e", font=("Helvetica", 11),
        ).pack(side="left", padx=8)

    # ── 길이 ──────────────────────────────────────────────────────
    section(win, "📏  경고 길이")
    length_var = tk.StringVar(value=OPTIONS.length)
    length_frame = tk.Frame(win, **FRAME_STYLE)
    length_frame.pack(anchor="w", padx=20)
    for opt in WarningOptions.LENGTH_OPTIONS:
        tk.Radiobutton(
            length_frame, text=opt, variable=length_var, value=opt,
            bg="#1e1e2e", fg="#89dceb", selectcolor="#313244",
            activebackground="#1e1e2e", font=("Helvetica", 11),
        ).pack(side="left", padx=8)

    # ── Gemini TTS 보이스 ──────────────────────────────────────────
    section(win, "🎙️  Gemini TTS 보이스")
    # 모델 고정 안내
    tk.Label(win, text="  gemini-2.5-flash-tts-preview (고정)",
             bg="#1e1e2e", fg="#45475a", font=("Helvetica", 10)).pack(anchor="w", padx=20)
    voice_var = tk.StringVar(value=OPTIONS.voice)
    voice_frame = tk.Frame(win, **FRAME_STYLE)
    voice_frame.pack(anchor="w", padx=20, pady=(4, 0))
    for i, (voice_name, voice_desc) in enumerate(WarningOptions.VOICE_OPTIONS.items()):
        col = i % 3
        row = i // 3
        rb = tk.Radiobutton(
            voice_frame,
            text=f"{voice_name}\n({voice_desc})",
            variable=voice_var, value=voice_name,
            bg="#1e1e2e", fg="#89dceb", selectcolor="#313244",
            activebackground="#1e1e2e", font=("Helvetica", 10),
            justify="center",
        )
        rb.grid(row=row, column=col, padx=10, pady=3, sticky="w")

    # ── 볼륨 슬라이더 ─────────────────────────────────────────────
    section(win, "🔊  볼륨")
    vol_var = tk.IntVar(value=OPTIONS.volume)
    vol_frame = tk.Frame(win, **FRAME_STYLE)
    vol_frame.pack(anchor="w", padx=20, fill="x")
    tk.Label(vol_frame, text="작게", bg="#1e1e2e", fg="#6c7086", font=("Helvetica", 10)).pack(side="left")
    tk.Scale(
        vol_frame, from_=0, to=100, orient="horizontal",
        variable=vol_var, length=220,
        bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
        highlightthickness=0, showvalue=True,
    ).pack(side="left", padx=6)
    tk.Label(vol_frame, text="크게", bg="#1e1e2e", fg="#6c7086", font=("Helvetica", 10)).pack(side="left")

    # ── 저장 버튼 ─────────────────────────────────────────────────
    def save_and_close():
        OPTIONS.tone   = tone_var.get()
        OPTIONS.length = length_var.get()
        OPTIONS.voice  = voice_var.get()
        OPTIONS.volume = vol_var.get()
        print(f"\n✅ 옵션 저장 완료: {OPTIONS.summary()}")
        win.destroy()

    tk.Button(
        win, text="  저장하기  ", command=save_and_close,
        bg="#89b4fa", fg="#1e1e2e", font=("Helvetica", 12, "bold"),
        relief="flat", cursor="hand2", padx=10, pady=6,
    ).pack(pady=18)

    win.mainloop()


# =========================
# 프롬프트 빌더 (옵션 반영)
# =========================
def _build_system_prompt() -> str:
    """옵션에 따라 SYSTEM_PROMPT 동적 생성"""
    tone_desc = {
        "부드러운": "자연스럽고 친근한 말투로",
        "중간":     "적절히 단호하지만 위협적이지 않은 말투로",
        "단호":     "매우 단호하고 간결한 명령형 말투로",
    }.get(OPTIONS.tone, "적절한 말투로")

    length_desc = {
        "단문": "1문장으로",
        "중문": "1~2문장으로",
        "장문": "2~3문장으로 위험 이유와 행동 방안을 함께",
    }.get(OPTIONS.length, "10~15단어로")

    # 프로파일 기반 추가 맥락
    age_desc = {
        "20대": "20대 운전자로, 자신감이 높고 위험을 과소평가하는 경향이 있습니다",
        "30대": "30대 운전자입니다",
        "40대": "40대 운전자로, 바쁜 일상 속 습관적 휴대폰 사용 가능성이 있습니다",
        "50대+": "50대 이상 운전자로, 친근하고 명확한 안내가 효과적입니다",
    }.get(OPTIONS.age, "")

    style_desc = {
        "Active": "운전 스타일이 적극적이라 강한 자극에 반응합니다",
        "Normal": "일반적인 운전 스타일입니다",
        "Comfort": "안전 위주의 운전 스타일이라 부드러운 안내가 효과적입니다",
    }.get(OPTIONS.driveStyle, "")

    exp_desc = {
        "입문": "운전 경험이 적어 구체적이고 친절한 행동 지침이 필요합니다",
        "일반": "평균적인 운전 경험을 보유하고 있습니다",
        "숙련": "운전 경험이 풍부하여 간결하고 핵심만 짚는 경고가 효과적입니다",
    }.get(OPTIONS.experience, "")

    return (
        f"당신은 차량 내 스마트 AI 비서입니다. 운전자의 위험 행동에 대해 경고 문장을 생성합니다. "
        f"문장은 {length_desc} 구성하세요. "
        f"{tone_desc} 작성하세요. "
        f"운전자 정보: {age_desc}. {style_desc}. {exp_desc}. "
        "이 정보를 바탕으로 해당 운전자에게 가장 효과적인 방식으로 경고 문장을 작성하세요. "
        "경고 메시지는 반드시 두 파트로 구성합니다: "
        "① 위험 상황 인식 (현재 상태를 짧게 언급) + ② 구체적 후속 행동 제안. "
        "예시: '휴대폰 사용 중 사고 위험이 높습니다, 지금 거치대에 올려두세요.'"
    )


# 경고 단계별 프롬프트 (에스컬레이션은 단계가 올라갈수록 자동으로 강도 상승)
USER_PROMPT_WARNING = {
    1: (
        "운전자가 휴대폰을 조작하고 있습니다. "
        "부드럽게 위험을 알리고, 지금 바로 할 수 있는 구체적인 행동을 함께 제안하는 "
        "한국어 경고 문장 1개를 생성하세요."
    ),
    2: (
        "운전자가 경고 후에도 계속 휴대폰을 조작하고 있습니다. "
        "더 단호한 톤으로 위험을 강조하고, 즉각 취해야 할 구체적인 행동을 명확히 지시하는 "
        "한국어 경고 문장 1개를 생성하세요."
    ),
    3: (
        "운전자가 두 번의 경고에도 휴대폰 조작을 멈추지 않습니다. "
        "매우 긴박하고 강한 톤으로, 지금 즉시 멈춰야 한다는 것과 취해야 할 행동을 포함한 "
        "한국어 경고 문장 1개를 생성하세요."
    ),
}

USER_PROMPT_POSITIVE = (
    "운전자가 경고를 듣고 휴대폰을 내려놓았습니다. "
    "안전한 선택을 칭찬하고, 앞으로도 안전 운전을 유지할 수 있도록 격려하는 "
    "따뜻하고 짧은 한국어 문장 1개를 생성하세요."
)

# 백업 문장 (단계별)
BACKUP_WARNING = {
    1: [
        "휴대폰 사용은 사고 위험을 높입니다, 잠깐 거치대에 올려두세요.",
        "운전 중 휴대폰 조작은 위험합니다, 지금 내려놓고 전방을 보세요.",
        "주의가 분산되고 있습니다, 휴대폰을 내려놓고 운전에 집중하세요.",
    ],
    2: [
        "계속된 휴대폰 사용은 매우 위험합니다, 즉시 내려놓고 전방을 주시하세요.",
        "경고합니다, 지금 당장 휴대폰을 내려놓고 두 손으로 운전하세요.",
        "심각한 사고 위험 상황입니다, 즉각 휴대폰을 놓고 전방을 보세요.",
    ],
    3: [
        "즉시 멈추세요! 휴대폰을 내려놓고 지금 당장 전방을 주시하세요!",
        "위험합니다! 지금 즉시 휴대폰을 좌석에 내려놓으세요!",
        "긴급 경고! 휴대폰을 놓고 두 손으로 핸들을 잡으세요!",
    ],
}

BACKUP_POSITIVE = [
    "잘 하셨어요, 이렇게 전방을 주시하며 안전 운전 계속해 주세요.",
    "훌륭합니다! 휴대폰은 도착 후 확인하고 지금처럼 운전에 집중해 주세요.",
    "좋아요, 안전한 선택이었습니다. 계속 이렇게 운전해 주세요.",
]

# 중복 방지 캐시
RECENT_CACHE = []
CACHE_MAX = 8


# =========================
# HitL 상태 관리
# =========================
class WarningState:
    """휴대폰 감지 및 HitL 피드백 루프 상태 머신"""

    IDLE        = "IDLE"
    DETECTING   = "DETECTING"
    WARNING     = "WARNING"
    IMPROVED    = "IMPROVED"
    DONE        = "DONE"

    DETECT_THRESHOLD_SEC   = 2.0
    IMPROVE_WINDOW_SEC     = 5.0
    MAX_WARNING_COUNT      = 3
    POST_ACTION_COOLDOWN   = 8.0
    IMPROVE_CONFIRM_FRAMES = 8

    # ★ 긍정 발화 딜레이 (초) — 휴대폰 내려놓은 후 이 시간만큼 기다렸다가 발화
    POSITIVE_DELAY_SEC = 3.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.state               = self.IDLE
        self.detect_start_time   = None
        self.warning_issued_time = None
        self.warning_count       = 0
        self.action_done_time    = None
        self.no_detect_frames    = 0
        self.improved_at         = None   # ★ 개선 감지 시각 저장용

    def is_in_cooldown(self, now: float) -> bool:
        if self.action_done_time is None:
            return False
        return (now - self.action_done_time) < self.POST_ACTION_COOLDOWN

    def update(self, phone_detected: bool, now: float):
        """
        반환값: ("action", payload)
            action = None | "warn" | "positive" | "log_done"
        """

        # ── 쿨다운 ────────────────────────────────────────────────
        if self.state in (self.IMPROVED, self.DONE):
            if self.is_in_cooldown(now):
                return None, None
            else:
                self.reset()

        # ── IDLE ──────────────────────────────────────────────────
        if self.state == self.IDLE:
            if phone_detected:
                self.detect_start_time = now
                self.state = self.DETECTING
            return None, None

        # ── DETECTING ─────────────────────────────────────────────
        if self.state == self.DETECTING:
            if not phone_detected:
                self.reset()
                return None, None
            elapsed = now - self.detect_start_time
            if elapsed >= OPTIONS.threshold:
                self.warning_count += 1
                self.warning_issued_time = now
                self.state = self.WARNING
                return "warn", self.warning_count
            return None, None

        # ── WARNING ───────────────────────────────────────────────
        if self.state == self.WARNING:
            window_elapsed = now - self.warning_issued_time

            if not phone_detected:
                self.no_detect_frames += 1

                # ★ 연속 미감지 확인 후 POSITIVE_DELAY_SEC 대기
                if self.no_detect_frames >= self.IMPROVE_CONFIRM_FRAMES:  # 8프레임 고정
                    if self.improved_at is None:
                        # 처음 개선 확인된 시각 기록
                        self.improved_at = now

                    waited = now - self.improved_at
                    if waited >= OPTIONS.delay:
                        # 딜레이 경과 → 긍정 발화
                        self.state = self.IMPROVED
                        self.action_done_time = now
                        return "positive", None

                return None, None
            else:
                self.no_detect_frames = 0
                self.improved_at = None  # 다시 감지되면 딜레이 타이머 리셋

            if window_elapsed >= self.IMPROVE_WINDOW_SEC:  # 5초 고정
                if self.warning_count >= self.MAX_WARNING_COUNT:  # 3회 고정
                    self.state = self.DONE
                    self.action_done_time = now
                    return "log_done", self.warning_count
                else:
                    self.warning_count += 1
                    self.warning_issued_time = now
                    self.no_detect_frames = 0
                    self.improved_at = None
                    return "warn", self.warning_count

            return None, None

        return None, None


# =========================
# 문장 후처리 / 중복 제어
# =========================
def _dedup(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def _post_one_sentence(text: str) -> str:
    text = _dedup(text)
    parts = re.split(r"(?<=[\.!?…])\s+", text)
    first = parts[0] if parts else text
    return first[:120]

def _not_duplicate(text: str) -> bool:
    return _dedup(text) not in RECENT_CACHE

def _remember(text: str):
    t = _dedup(text)
    if t:
        RECENT_CACHE.append(t)
        if len(RECENT_CACHE) > CACHE_MAX:
            del RECENT_CACHE[0]

def backup_utterance(pool: list) -> str:
    for _ in range(5):
        cand = random.choice(pool)
        if _not_duplicate(cand):
            return cand
    return pool[0]


# =========================
# LLM 호출
# =========================
def generate_warning(level: int) -> str:
    prompt = USER_PROMPT_WARNING.get(level, USER_PROMPT_WARNING[3])
    backup_pool = BACKUP_WARNING.get(level, BACKUP_WARNING[3])
    try:
        _cli = _get_gemini_client()
        if _cli is None:
            raise Exception("Gemini 클라이언트 없음")
        from google.genai import types as _gt
        resp = _cli.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=_gt.GenerateContentConfig(
                system_instruction=_build_system_prompt(),
                max_output_tokens=80,
            ),
        )
        text = resp.text if resp and resp.text else ""
        text = _post_one_sentence(text)
        if not text or not _not_duplicate(text):
            text = backup_utterance(backup_pool)
        _remember(text)
        return text
    except Exception as e:
        print(f"⚠️ LLM 실패 (경고 level={level}): {e}")
        text = backup_utterance(backup_pool)
        _remember(text)
        return text


def generate_positive() -> str:
    try:
        _cli = _get_gemini_client()
        if _cli is None:
            raise Exception("Gemini 클라이언트 없음")
        from google.genai import types as _gt
        resp = _cli.models.generate_content(
            model=LLM_MODEL,
            contents=USER_PROMPT_POSITIVE,
            config=_gt.GenerateContentConfig(
                system_instruction=_build_system_prompt(),
                max_output_tokens=60,
            ),
        )
        text = resp.text if resp and resp.text else ""
        text = _post_one_sentence(text)
        if not text or not _not_duplicate(text):
            text = backup_utterance(BACKUP_POSITIVE)
        _remember(text)
        return text
    except Exception as e:
        print(f"⚠️ LLM 실패 (긍정 강화): {e}")
        text = backup_utterance(BACKUP_POSITIVE)
        _remember(text)
        return text


# =========================
# 오디오 (Gemini TTS + macOS say 폴백)
# =========================
def play_alert_sound():
    def _play():
        try:
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"경고음 재생 실패: {e}")
    threading.Thread(target=_play, daemon=True).start()


def _speak_gemini(text: str, voice: str, volume: int):
    """Gemini TTS로 발화. 실패 시 False 반환."""
    gemini = _get_gemini_client()
    if not gemini:
        return False
    try:
        resp = gemini.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice,
                        )
                    )
                ),
            ),
        )
        audio_data = resp.candidates[0].content.parts[0].inline_data.data

        # pyaudio로 직접 스트리밍 재생 (파일 저장 없음)
        try:
            import pyaudio
            vol_factor = max(0.0, min(1.0, volume / 100.0))
            # 볼륨 적용 (PCM int16 스케일링)
            import numpy as np
            samples = np.frombuffer(audio_data, dtype=np.int16)
            samples = (samples * vol_factor).astype(np.int16)

            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True,
            )
            stream.write(samples.tobytes())
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except ImportError:
            # pyaudio 없으면 wav 저장 폴백
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                with wave.open(tmp_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(audio_data)
            vol_factor = max(0.0, min(1.0, volume / 100.0))
            subprocess.run(["afplay", "-v", str(vol_factor), tmp_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.unlink(tmp_path)
        return True

    except Exception as e:
        print(f"⚠️ Gemini TTS 실패: {e}")
        return False


def _speak_fallback(text: str):
    """Gemini TTS 실패 시 macOS say 명령어로 폴백."""
    try:
        os.system(f"say -v Yuna '{text}'")
    except Exception as e:
        print(f"⚠️ say 폴백 실패: {e}")


def speak(text: str, voice: str = None, volume: int = None):
    """
    Gemini TTS로 발화.
    - GEMINI_API_KEY 없거나 API 실패 시 macOS say 로 자동 폴백
    - voice  : None이면 OPTIONS.voice 사용
    - volume : None이면 OPTIONS.volume 사용
    """
    _voice  = voice  if voice  is not None else OPTIONS.get_voice()
    _volume = volume if volume is not None else OPTIONS.volume

    # 경고음 먼저 (비동기)
    def _double_alert():
        play_alert_sound()
        time.sleep(0.2)
        play_alert_sound()
    threading.Thread(target=_double_alert, daemon=True).start()
    time.sleep(0.3)

    # Gemini TTS 시도 → 클라이언트 없으면 macOS say 폴백
    if _get_gemini_client() is None:
        print("ℹ️  Gemini 클라이언트 없음 → macOS say 폴백")
        _speak_fallback(text)
        return

    success = _speak_gemini(text, _voice, _volume)
    if not success:
        print("ℹ️  Gemini TTS 실패 → macOS say 폴백")
        _speak_fallback(text)


# =========================
# 시각화
# =========================
def visualize(image, detection_result) -> np.ndarray:
    TEXT_COLOR = (0, 255, 0)
    vis_image = np.copy(image)
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        cv2.rectangle(vis_image,
                      (bbox.origin_x, bbox.origin_y),
                      (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                      TEXT_COLOR, 3)
        cat = detection.categories[0]
        label = f"{cat.category_name} ({cat.score:.2f})"
        pos = (10 + bbox.origin_x, bbox.origin_y - 10)
        if pos[1] < 10:
            pos = (10 + bbox.origin_x, bbox.origin_y + 20)
        cv2.putText(vis_image, label, pos,
                    cv2.FONT_HERSHEY_PLAIN, 1, TEXT_COLOR, 1)
    return vis_image


# =========================
# 메인 루프
# =========================
def run(model_path: str, camera_id: int, width: int, height: int):
    fps_avg_frame_count = 10
    counter = 0
    fps = 0.0
    fps_timer = time.time()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit(f"ERROR: 웹캠 열기 실패 (카메라 ID {camera_id})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detection_result_list = []

    def visualize_callback(result: vision.ObjectDetectorResult,
                           output_image: mp.Image, timestamp_ms: int):
        result.timestamp_ms = timestamp_ms
        detection_result_list.append(result)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        score_threshold=0.5,
        result_callback=visualize_callback,
    )
    try:
        detector = vision.ObjectDetector.create_from_options(options)
    except Exception as e:
        sys.exit(f"ERROR: MediaPipe 초기화 실패: {e}")

    ws = WarningState()
    last_phone_detected = False  # 직전 프레임 감지 상태 유지용

    print("=== HitL DMS 시작 ===")
    print(f"  {OPTIONS.summary()}")
    print(f"  - 감지 임계값 : {WarningState.DETECT_THRESHOLD_SEC}초 지속 감지")
    print(f"  - 개선 판단   : 경고 후 {WarningState.IMPROVE_WINDOW_SEC}초 내 연속 {WarningState.IMPROVE_CONFIRM_FRAMES}프레임 미감지")
    print(f"  - 긍정 발화 딜레이 : {WarningState.POSITIVE_DELAY_SEC}초")
    print(f"  - 최대 경고   : {WarningState.MAX_WARNING_COUNT}회")
    print("  - ESC 키로 종료\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("ERROR: 프레임 읽기 실패.")
            break

        counter += 1
        frame = cv2.flip(frame, 1)
        now = time.time()

        # ── 탐지 ──────────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detector.detect_async(mp_image, counter)
        current_frame = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)

        # FPS
        if counter % fps_avg_frame_count == 0:
            fps = fps_avg_frame_count / (now - fps_timer)
            fps_timer = now
        # FPS는 터미널 출력으로 대체됨

        # ── 결과 처리 ─────────────────────────────────────────────
        # 결과가 없는 프레임은 직전 감지 상태 유지 (비동기 누락 방지)
        if detection_result_list:
            result = detection_result_list.pop(0)
            phone_detected_this_frame = False

            detected_labels = [d.categories[0].category_name for d in result.detections]
            if counter % 30 == 0 and detected_labels:
                print(f"\n🔍 감지됨: {detected_labels}", flush=True)

            for det in result.detections:
                if det.categories[0].category_name == "cell phone":
                    phone_detected_this_frame = True
                    break
            last_phone_detected = phone_detected_this_frame
        else:
            # 결과 없는 프레임 → 직전 상태 유지
            phone_detected_this_frame = last_phone_detected

            # 상태 터미널 출력 (1초마다)
            if counter % 15 == 0:
                status_text = "MONITORING"
                if ws.state == WarningState.DETECTING:
                    elapsed_detect = now - ws.detect_start_time
                    status_text = f"DETECTING {elapsed_detect:.1f}s / {OPTIONS.threshold}s"
                elif ws.state == WarningState.WARNING:
                    elapsed_window = now - ws.warning_issued_time
                    remain = max(0, WarningState.IMPROVE_WINDOW_SEC - elapsed_window)
                    if ws.improved_at is not None:
                        delay_remain = max(0, OPTIONS.delay - (now - ws.improved_at))
                        status_text = f"IMPROVED — 칭찬 대기 {delay_remain:.1f}s"
                    else:
                        status_text = f"WARNING #{ws.warning_count} | {remain:.1f}s 남음"
                elif ws.state in (WarningState.IMPROVED, WarningState.DONE):
                    status_text = "COOLDOWN"
                print(f"\r📷 [{status_text}] FPS:{fps:.0f} | {OPTIONS.tone}/{OPTIONS.length}/{OPTIONS.voice}", end="", flush=True)

        # ── HitL 상태 머신 업데이트 ───────────────────────────────
        action, payload = ws.update(phone_detected_this_frame, now)

        if action == "warn":
            level = payload
            alert_queue.put({'type': 'warn', 'level': level})
            print(f"\n📱 [경고 #{level}] 휴대폰 {WarningState.DETECT_THRESHOLD_SEC}초 이상 감지")
            print(f"   {OPTIONS.summary()}")
            text = generate_warning(level)
            print(f"🗣️  {text}")
            threading.Thread(target=speak, args=(text,), daemon=True).start()

        elif action == "positive":
            alert_queue.put({'type': 'positive'})
            print(f"\n✅ [개선 감지] {WarningState.POSITIVE_DELAY_SEC}초 딜레이 후 긍정 강화 발화")
            text = generate_positive()
            print(f"🗣️  {text}")
            # 긍정 발화 (볼륨 살짝 낮춰서 부드럽게)
            gentle_vol = max(OPTIONS.volume - 15, 20)
            threading.Thread(target=speak, args=(text, None, gentle_vol), daemon=True).start()

        elif action == "log_done":
            print(f"\n⛔ [루프 종료] 최대 {WarningState.MAX_WARNING_COUNT}회 경고 후에도 미개선 → 로그 기록")

        # 1ms 슬립으로 CPU 과부하 방지 (ESC는 Ctrl+C로 대체)
        time.sleep(0.001)

    detector.close()
    cap.release()
    print("\n카메라 종료")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="HitL DMS: 휴대폰 감지 → 피드백 루프 경고 시스템"
    )
    parser.add_argument("--model",       default="efficientdet_lite0.tflite")
    parser.add_argument("--cameraId",    type=int, default=0)
    parser.add_argument("--frameWidth",  type=int, default=640)
    parser.add_argument("--frameHeight", type=int, default=480)
    parser.add_argument("--no-options",  action="store_true",
                        help="옵션 설정 창 건너뛰고 기본값으로 바로 시작")
    args = parser.parse_args()

    # ★ Flask 서버를 백그라운드 스레드로 먼저 시작
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("🌐 Flask 설정 서버 시작: http://127.0.0.1:5050")
    print("   HTML에서 저장하기 버튼 클릭 시 OPTIONS 실시간 반영")

    # ★ 옵션 설정 창 (--no-options 플래그로 스킵 가능)
    if not args.no_options:
        print("⚙️  옵션 설정 창을 여는 중... (닫으면 기본값으로 시작)")
        open_options_window()

    run(args.model, args.cameraId, args.frameWidth, args.frameHeight)


if __name__ == "__main__":
    main()
