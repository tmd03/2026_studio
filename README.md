# 🚗 AI-Powered Driver Monitoring System
> Real-time driver distraction detection and personalized voice intervention using MediaPipe, Gemini, and Human-in-the-Loop feedback.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Computer%20Vision-orange)
![Gemini](https://img.shields.io/badge/Gemini-LLM%20%2B%20TTS-purple)
![Flask](https://img.shields.io/badge/Flask-Web%20Server-black)

---

## 📖 Overview

This project presents an AI-powered Driver Monitoring System (DMS) that detects distracted driving behavior in real time and provides personalized voice-based interventions.

Unlike traditional warning systems that simply trigger alarms, this system generates context-aware warnings using Large Language Models (LLMs) and adapts its feedback according to driver behavior and profile information.

### Core Components

- 📷 Real-time mobile phone detection using MediaPipe
- 🤖 Gemini-based warning sentence generation
- 🎙️ Gemini TTS voice feedback
- 👤 Personalized driver profiles
- 🔄 Human-in-the-Loop feedback mechanism
- 🌐 Web-based configuration interface

---

## ✨ Features

### 📱 Real-Time Driver Monitoring

The system continuously monitors the driver through a webcam and detects mobile phone usage during driving.

```text
Webcam
   ↓
MediaPipe Detection
   ↓
Phone Usage Recognition
```

---

### 🧠 Personalized AI Warning Generation

Warning messages are dynamically generated based on:

- Driver age
- Driving experience
- Driving style
- Preferred warning tone
- Preferred warning length

Example:

> "휴대폰 사용 중 사고 위험이 높습니다. 지금 거치대에 올려두세요."

---

### 🔊 AI Voice Feedback

Generated warnings are delivered through Gemini TTS.

#### Available Voice Profiles

| Voice | Style |
|---------|---------|
| Kore | Firm |
| Puck | Energetic |
| Charon | Informative |
| Aoede | Bright |
| Achird | Friendly |
| Sulafat | Warm |

Additional features:

- Adjustable volume
- Multiple voice personas
- macOS speech fallback
- Real-time voice preview

---

### 🔄 Human-in-the-Loop Escalation

If unsafe behavior continues, warning intensity increases automatically.

```text
Detection
    ↓
Warning Level 1
    ↓
Warning Level 2
    ↓
Warning Level 3
    ↓
Behavior Improved?
   ├─ Yes → Positive Reinforcement
   └─ No  → Incident Logging
```

---

### 🌐 Real-Time Configuration Dashboard

Users can configure:

- Warning tone
- Warning length
- Voice type
- Volume
- Driver age group
- Driving style
- Driving experience

through a browser-based interface connected directly to the Python backend.

---

## 🏗 System Architecture

```text
┌─────────────┐
│ Webcam Feed │
└──────┬──────┘
       │
       ▼
┌───────────────────┐
│ MediaPipe Detector│
└──────┬────────────┘
       │
       ▼
┌───────────────────┐
│ HitL State Engine │
└──────┬────────────┘
       │
       ▼
┌───────────────────┐
│ Gemini LLM        │
│ Warning Generator │
└──────┬────────────┘
       │
       ▼
┌───────────────────┐
│ Gemini TTS Engine │
└──────┬────────────┘
       │
       ▼
   Voice Feedback
```

---

## 🛠 Technology Stack

### Frontend

- HTML
- CSS
- JavaScript

### Backend

- Python
- Flask
- Flask-CORS

### AI & Computer Vision

- MediaPipe Object Detection
- Gemini 3.1 Flash
- Gemini 2.5 flash TTS


---

## 🚀 Run

### Create Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

### Launch System

```bash
python3 2026_1_studio_v3.py
```

### Skip Configuration Window

```bash
python3 2026_1_studio_v3.py --no-options
```

---

## 📂 Project Structure

```text
.
├── 2026_1_studio_v3.py
├── studio_v2(0608).html
├── efficientdet_lite0.tflite
├── requirements.txt
├── .env
└── README.md
```

---

## 🧪 Example Workflow

```text
Driver Uses Phone
         ↓
MediaPipe Detects Phone
         ↓
HitL State Machine Triggered
         ↓
Gemini Generates Warning
         ↓
Gemini TTS Speaks Warning
         ↓
Driver Stops Behavior
         ↓
Positive Reinforcement Message
```

---

## 🎯 Research Motivation

Traditional Driver Monitoring Systems mainly rely on generic warning sounds.

This project investigates whether:

> Context-aware AI-generated explanations can improve driver understanding, compliance, and user experience compared to conventional alert-based systems.

The system serves as a research prototype for studying proactive AI interventions in future automotive environments.

---

## 📸 Demo

### Warning Example

```text
⚠️ 휴대폰 사용은 사고 위험을 높입니다.
지금 거치대에 올려두고 전방을 주시하세요.
```

### Positive Feedback Example

```text
✅ 잘 하셨어요.
계속 전방을 주시하며 안전 운전을 이어가세요.
```

---

## 📄 License

This project is intended for research and educational purposes.
