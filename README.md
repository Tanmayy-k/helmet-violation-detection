# 🪖 Smart Helmet Violation Detection System
### AI-Based Campus Safety Enforcement | Computer Vision Project

![Python](https://img.shields.io/badge/Python-3.14-blue)
![YOLOv5](https://img.shields.io/badge/YOLOv5-Custom%20Trained-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Number%20Plate-orange)

---

## 📌 Problem Statement
College campuses face challenges enforcing helmet rules at entry gates.
Manual checking is inefficient, inconsistent, and requires dedicated staff.
This system automates the entire process using AI and computer vision.

---

## 💡 Solution
A real-time AI system placed at the college entry gate that:
- Detects riders without helmets automatically
- Captures violation images with timestamp
- Reads number plate using OCR
- Logs all violations for admin review

---

## 🎯 Features
| Feature | Status |
|--------|--------|
| Real-time helmet detection | ✅ Done |
| No-helmet violation trigger | ✅ Done |
| Violation image capture | ✅ Done |
| Number plate detection | ✅ Done |
| OCR plate text reading | ✅ Done |
| 10 second cooldown timer | ✅ Done |
| Live violation counter on screen | ✅ Done |
| Red border alert on violation | ✅ Done |
| Beep sound on violation | ✅ Done |
| Confidence score logging | ✅ Done |
| Hourly violation stats | ✅ Done |
| Web dashboard |✅  Done |

---

## 🧠 AI Model
- Architecture: YOLOv5s
- Confidence Threshold: 0.65
- NMS IoU Threshold: 0.45
- Classes Detected:
  - `0` → Helmet
  - `1` → Human Head
  - `2` → Motorcycle
  - `3` → Vehicle Registration Plate

---

## 🛠️ Tech Stack
| Category | Technology |
|----------|-----------|
| AI Model | YOLOv5 |
| Language | Python 3.14 |
| Computer Vision | OpenCV |
| OCR | EasyOCR |
| Deep Learning | PyTorch |
| Web Dashboard | Flask + HTML/CSS |

---

## 📁 Project Structure
```
helmet-violation-detection/
│
├── yolov5.py              # Main detection script
├── best.pt                # Trained YOLOv5 model weights
├── helmet_plate.yaml      # Dataset configuration
├── requirements.txt       # Python dependencies
│
├── violations/            # Auto-generated violation images
│   └── violation_YYYY-MM-DD_HH-MM-SS.jpg
│
├── violations_log.csv     # Violation log
│   └── timestamp, plate_text, confidence, image_path
│
├── hourly_stats.csv       # Hourly violation statistics
│   └── date, hour, count
│
├── data/                  # Training dataset
│   ├── train/             # 1538 training images
│   ├── test/              # 385 test images
│   └── images/            # 1923 validation images
│
└── templates/             # Frontend dashboard (in progress)
    └── index.html
```

---

## ⚙️ How It Works
```
Camera Feed
    ↓
Frame Extraction
    ↓
YOLOv5 Detection (helmet, head, motorcycle, plate)
    ↓
Violation Check (head detected without nearby helmet?)
    ↓
    ├── NO VIOLATION → Continue monitoring
    └── VIOLATION DETECTED
            ↓
        Red border + Beep alert
            ↓
        Capture frame → Save to violations/
            ↓
        Crop number plate → EasyOCR
            ↓
        Log to violations_log.csv + hourly_stats.csv
```

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Tanmayy-k/helmet-violation-detection
cd helmet-violation-detection
```

**2. Activate virtual environment**
```bash
.\venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run detection**
```bash
python yolov5.py
```

**5. Press `Q` to quit**

---

## 📊 Output Files

**violations_log.csv**
| timestamp | plate_text | confidence | image_path |
|-----------|-----------|------------|------------|
| 2024-03-12 14:30:22 | MH31AB1234 | 0.87 | violations/violation_... |

**hourly_stats.csv**
| date | hour | count |
|------|------|-------|
| 2024-03-12 | 08 | 5 |

---

## 🏫 Use Case
> Deployed at college entry gate to enforce helmet rules,
> reduce manual checking and improve campus safety.

Can be extended to:
- Traffic enforcement systems
- CCTV integration
- Automated fine generation
---

## 👥 Team
- Tanmay Kshirsagar
- Atharva Borate
- Khushi Bhadangkar
- Anushri Ghosh

---

## 📄 License
MIT License
