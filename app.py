"""
VisionPro — Professional Vision AI
Fixed detection algorithms:
  - Hands: higher area threshold + stricter finger counting to avoid false positives
  - Emotions: multi-pass analysis with confidence scoring
  - Objects: minimum coverage threshold + aspect-ratio stricter rules
  - Body: position-based actions

Run: python app.py
"""

import io, base64, time, cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from PIL import Image
from functools import wraps

app = Flask(__name__)
app.secret_key = "visionpro_secret_2024"
CORS(app)

# ── Demo users (in production use a database) ─────────────────────────────────
USERS = {
    "admin":  "admin123",
    "demo":   "demo123",
    "user":   "password",
}

# ── Load OpenCV cascades ──────────────────────────────────────────────────────
HAAR = cv2.data.haarcascades
face_cascade  = cv2.CascadeClassifier(HAAR + "haarcascade_frontalface_default.xml")
eye_cascade   = cv2.CascadeClassifier(HAAR + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(HAAR + "haarcascade_smile.xml")
body_cascade  = cv2.CascadeClassifier(HAAR + "haarcascade_fullbody.xml")
upper_cascade = cv2.CascadeClassifier(HAAR + "haarcascade_upperbody.xml")

print("[startup] OpenCV cascades loaded")


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def decode_image(b64):
    raw = base64.b64decode(b64.split(",")[-1])
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb, bgr


# ══════════════════════════════════════════════════════════════════════════════
#  FACE + EMOTION  —  uses 5-zone brightness analysis + eye/smile cascade
# ══════════════════════════════════════════════════════════════════════════════
def analyze_emotion(roi_gray, smile_count, eye_count):
    if roi_gray.size == 0 or roi_gray.shape[0] < 30 or roi_gray.shape[1] < 30:
        return "😐 Neutral", 70

    h, w = roi_gray.shape

    # Resize for consistent analysis
    face = cv2.resize(roi_gray, (64, 64))

    # Zones
    brow   = face[0:16,   8:56]    # forehead / brow area
    leye   = face[16:32,  4:28]    # left eye area
    reye   = face[16:32,  36:60]   # right eye area
    nose   = face[24:40,  24:40]   # nose bridge
    mouth  = face[40:60,  10:54]   # mouth area
    lcheck = face[20:50,  0:20]    # left cheek
    rcheck = face[20:50,  44:64]   # right cheek

    def mean(z): return float(np.mean(z)) if z.size > 0 else 128.0
    def std(z):  return float(np.std(z))  if z.size > 0 else 0.0

    brow_m   = mean(brow)
    leye_m   = mean(leye)
    reye_m   = mean(reye)
    mouth_m  = mean(mouth)
    lchk_m   = mean(lcheck)
    rchk_m   = mean(rcheck)
    face_std = std(face)

    # Derived features
    mouth_dark      = mouth_m < 95              # open mouth = dark inside
    eyes_open       = (leye_m + reye_m) / 2 > 110
    brows_raised    = brow_m > 148
    brows_furrowed  = brow_m < 95
    high_tension    = face_std > 52
    cheek_asymmetry = abs(lchk_m - rchk_m) > 18
    overall_dark    = mean(face) < 112

    # Priority order: most distinctive features first
    # LAUGHING: smile + mouth wide open + eyes narrowed
    if smile_count >= 1 and mouth_dark and leye_m < 115:
        return "😂 Laughing", 91

    # HAPPY: smile detected (primary signal)
    if smile_count >= 1:
        return "😊 Happy", min(93, 78 + smile_count * 5)

    # SURPRISED: mouth open + eyes wide + brows raised
    if mouth_dark and brows_raised and eyes_open and eye_count >= 1:
        return "😲 Surprised", 86

    # FEARFUL: mouth open + brows raised + high tension + eyes wide
    if mouth_dark and brows_raised and high_tension:
        return "😨 Fearful", 79

    # ANGRY: brows down/furrowed + high tension + mouth closed
    if brows_furrowed and high_tension and not mouth_dark:
        return "😠 Angry", 83

    # DISGUSTED: brows down + low tension + slight asymmetry
    if brows_furrowed and not high_tension and cheek_asymmetry:
        return "🤢 Disgusted", 74

    # SAD: overall dark face + no smile + brows in middle position
    if overall_dark and not mouth_dark and not brows_raised:
        return "😢 Sad", 76

    # CRYING: sad + mouth slightly open
    if overall_dark and mouth_dark and smile_count == 0:
        return "😭 Crying", 73

    # THINKING: asymmetric cheeks (looking sideways)
    if cheek_asymmetry and eye_count >= 1 and not mouth_dark:
        return "🤔 Thinking", 69

    return "😐 Neutral", 80


def detect_faces(bgr):
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
    )
    output = []
    for (x, y, fw, fh) in faces:
        roi_g  = gray[y:y+fh, x:x+fw]
        eyes   = eye_cascade.detectMultiScale(roi_g, 1.1, 5, minSize=(20,20))
        smiles = smile_cascade.detectMultiScale(roi_g, 1.7, 22, minSize=(25,25))
        emotion, conf = analyze_emotion(roi_g, len(smiles), len(eyes))
        output.append({
            "box":        [int(x), int(y), int(fw), int(fh)],
            "emotion":    emotion,
            "confidence": conf,
            "eyes":       f"{min(len(eyes), 2)} eye(s) visible",
        })
    return output


# ══════════════════════════════════════════════════════════════════════════════
#  HAND GESTURE  —  Strict skin detection + convexity defect finger count
#  Key fixes:
#    1. Minimum area 8000px (was 4000) to ignore small skin blobs
#    2. Only count defects with angle < 80° AND depth > 20px
#    3. Max 4 defect gaps = 5 fingers (not 6+)
#    4. Solidity threshold to distinguish fist from open
# ══════════════════════════════════════════════════════════════════════════════
def detect_hands(bgr):
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Two skin ranges merged
    m1 = cv2.inRange(hsv, np.array([0,   25, 80]),  np.array([17,  255, 255]))
    m2 = cv2.inRange(hsv, np.array([168, 25, 80]),  np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)

    # Morphological cleanup — more aggressive to remove noise
    k5   = np.ones((5,5), np.uint8)
    k9   = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k9)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output  = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)

        # STRICT: minimum 8000px, max 30% of frame
        if area < 8000 or area > (w * h * 0.30):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Skip face region (top-center of frame)
        cx_n = (x + bw/2) / w
        cy_n = (y + bh/2) / h

        # If region is in top 40% and center 40% = likely face, skip
        if cy_n < 0.40 and 0.30 < cx_n < 0.70:
            continue

        # Aspect ratio: hands are roughly square-ish or taller
        ar = bw / bh if bh > 0 else 1
        if ar > 2.0:   # too wide = probably arm, not hand
            continue

        hull = cv2.convexHull(cnt, returnPoints=False)

        # Solidity
        hull_pts  = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull_pts)
        solidity  = area / hull_area if hull_area > 0 else 1.0

        finger_count = 0

        try:
            if len(hull) > 3:
                defects = cv2.convexityDefects(cnt, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        depth = d / 256.0
                        if depth < 20:     # ignore shallow defects (noise)
                            continue
                        start = tuple(cnt[s][0])
                        end   = tuple(cnt[e][0])
                        far   = tuple(cnt[f][0])

                        a = np.array(start, dtype=np.float32) - np.array(far, dtype=np.float32)
                        b = np.array(end,   dtype=np.float32) - np.array(far, dtype=np.float32)
                        la, lb = np.linalg.norm(a), np.linalg.norm(b)
                        if la < 1 or lb < 1:
                            continue

                        cos_a = np.dot(a, b) / (la * lb)
                        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

                        # STRICT: only angles < 80° (real finger gaps)
                        if angle < 80:
                            finger_count += 1
        except Exception:
            pass

        # Cap at 4 gaps = 5 fingers
        finger_count = min(finger_count, 4)

        # Classify
        peri    = cv2.arcLength(cnt, True)
        circ    = 4 * np.pi * area / (peri * peri + 1e-6)

        if solidity > 0.88 and finger_count == 0:
            gesture, meaning = "✊ Fist",           "All fingers closed tightly"
        elif finger_count == 0 and solidity > 0.75:
            gesture, meaning = "✊ Closed Hand",    "Fingers together"
        elif finger_count == 0:
            gesture, meaning = "👍 Thumbs Up",      "Thumb extended"
        elif finger_count == 1:
            gesture, meaning = "✌️ Peace Sign",     "Two fingers raised"
        elif finger_count == 2:
            gesture, meaning = "🤟 Three Fingers",  "Three fingers raised"
        elif finger_count == 3:
            gesture, meaning = "4️⃣ Four Fingers",   "Four fingers raised"
        elif finger_count == 4:
            gesture, meaning = "✋ Open Hand",      "All five fingers extended"
        else:
            gesture, meaning = "✋ Hand",           "Hand detected"

        # Override: near-circular small blob = OK sign
        if circ > 0.72 and area < 15000:
            gesture, meaning = "👌 OK Sign",        "Thumb and index forming circle"

        output.append({
            "hand":     "Hand",
            "gesture":  gesture,
            "meaning":  meaning,
            "fingers":  finger_count + 1 if finger_count > 0 else (0 if solidity > 0.85 else 1),
            "box_norm": [x/w, y/h, bw/w, bh/h],
        })

    return output[:2]


# ══════════════════════════════════════════════════════════════════════════════
#  BODY POSE
# ══════════════════════════════════════════════════════════════════════════════
def detect_pose(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = bgr.shape[:2]

    bodies = body_cascade.detectMultiScale(gray, 1.05, 3, minSize=(60,120))
    upper  = upper_cascade.detectMultiScale(gray, 1.05, 3, minSize=(60,60))

    if len(bodies) == 0 and len(upper) == 0:
        return {"detected": False, "actions": [], "box_norm": []}

    boxes = list(bodies) if len(bodies) > 0 else list(upper)
    x, y, bw, bh = max(boxes, key=lambda b: b[2]*b[3])
    ar = bw / bh if bh > 0 else 1

    actions = []
    cx = (x + bw/2) / w
    coverage = (bw * bh) / (w * h)

    if len(bodies) > 0:
        if ar < 0.45:   actions.append({"action":"🧍 Standing",     "detail":"Full upright body"})
        elif ar > 0.75: actions.append({"action":"🪑 Sitting",      "detail":"Body wider — seated posture"})
        else:           actions.append({"action":"🧍 Person",       "detail":"Person in frame"})
    else:
        actions.append({"action":"👤 Upper Body",    "detail":"Torso and head visible"})

    if cx < 0.35:   actions.append({"action":"↙️ Left of Frame",  "detail":"Person on left side"})
    elif cx > 0.65: actions.append({"action":"↘️ Right of Frame", "detail":"Person on right side"})
    else:           actions.append({"action":"🎯 Center Frame",   "detail":"Person centered"})

    if coverage > 0.35: actions.append({"action":"🔍 Close Up",    "detail":"Very close to camera"})
    elif coverage < 0.06: actions.append({"action":"📏 Far Away",  "detail":"Far from camera"})

    if y / h < 0.08:    actions.append({"action":"🦘 Jumping",    "detail":"Body near top of frame"})

    return {"detected": True, "actions": actions,
            "box_norm": [x/w, y/h, bw/w, bh/h]}


# ══════════════════════════════════════════════════════════════════════════════
#  OBJECT DETECTION
#  Key fix: minimum area 1.5% of frame + stricter shape/color mapping
#  A white rectangular object is NOT automatically a cup/mug
#  — it needs to be tall+narrow to be a cup
# ══════════════════════════════════════════════════════════════════════════════
def identify_object(color, shape, ar, area_pct, bh, H):
    c       = color.lower()
    is_tall = ar < 0.60
    is_wide = ar > 1.60
    is_sq   = 0.80 < ar < 1.20
    is_circ = shape == "Circle"
    is_rect = shape in ("Rectangle","Square")
    big     = area_pct > 14
    med     = 3 < area_pct <= 14
    small   = area_pct <= 3

    # Height proportion relative to frame
    tall_in_frame = (bh / H) > 0.35

    if "red" in c:
        if is_circ and small:       return "🍎 Apple",             87, "Small round red fruit"
        if is_circ and med:         return "🍅 Tomato",             82, "Medium round red vegetable"
        if is_tall and tall_in_frame: return "🥤 Red Can/Bottle",  78, "Tall red container"
        if is_rect and is_wide:     return "📕 Red Book",           73, "Flat rectangular red object"
        if is_sq and big:           return "🎁 Red Box",            70, "Square red box"
        return                             "🔴 Red Object",         62, "Red colored object"

    if "orange" in c:
        if is_circ and big:         return "🏀 Basketball",         88, "Large round orange ball"
        if is_circ:                 return "🍊 Orange",             87, "Round orange citrus fruit"
        if is_rect:                 return "📙 Orange Book",        70, "Rectangular orange object"
        return                             "🟠 Orange Object",      60, "Orange colored object"

    if "yellow" in c:
        if is_circ and small:       return "🍋 Lemon",              84, "Small round yellow fruit"
        if is_circ:                 return "🟡 Yellow Ball",         78, "Round yellow ball"
        if is_tall and small:       return "🍌 Banana",             80, "Narrow tall yellow fruit"
        if is_rect and is_wide:     return "📒 Yellow Notepad",     71, "Wide yellow book/pad"
        if is_sq and big:           return "📦 Yellow Box",         69, "Large yellow box"
        return                             "🟡 Yellow Object",      60, "Yellow colored object"

    if "green" in c:
        if is_circ and small:       return "🍏 Green Apple",        87, "Small round green fruit"
        if is_tall and tall_in_frame: return "🍾 Green Bottle",     79, "Tall green bottle"
        if is_tall and big:         return "🌵 Plant",              70, "Tall green plant"
        if is_rect:                 return "📗 Green Book",         73, "Rectangular green object"
        return                             "🟢 Green Object",       60, "Green colored object"

    if "blue" in c:
        if is_circ:                 return "🔵 Blue Ball",          78, "Round blue ball"
        if is_tall and tall_in_frame: return "🧴 Blue Bottle",      78, "Tall blue bottle/container"
        if is_rect and big:         return "🖥️ Screen/Display",    68, "Large blue rectangular screen"
        if is_rect and small:       return "💳 Blue Card",          73, "Small blue card"
        if is_rect:                 return "📘 Blue Book",          73, "Rectangular blue object"
        return                             "🔵 Blue Object",        60, "Blue colored object"

    if "purple" in c:
        if is_circ:                 return "🍇 Grapes",             76, "Round purple fruit cluster"
        if is_rect:                 return "📓 Purple Notebook",    70, "Rectangular purple object"
        return                             "🟣 Purple Object",      58, "Purple colored object"

    if "white" in c:
        # FIXED: white paper = wide+large, cup = tall+narrow+small, NOT default
        if is_wide and big:         return "📄 Paper Sheet",        80, "Large white flat paper"
        if is_wide and med:         return "📋 Document/Paper",     76, "White paper or document"
        if is_tall and small and tall_in_frame:
                                    return "☕ Cup/Mug",            72, "Tall narrow white cup"
        if is_sq and small:         return "🪄 White Object",       60, "Small square white object"
        if is_circ:                 return "⚾ White Ball",         74, "Round white ball"
        if is_rect and med:         return "📄 White Paper",        70, "White rectangular sheet"
        return                             "⚪ White Object",       54, "White colored object"

    if "black" in c:
        if is_rect and small:       return "📱 Phone",              83, "Small black rectangle — phone"
        if is_rect and big:         return "💻 Laptop/Monitor",     76, "Large black screen"
        if is_circ:                 return "⚫ Black Ball",         70, "Round black object"
        return                             "⬛ Black Object",       55, "Black colored object"

    return "❓ Unknown Object", 48, f"{color} {shape}"


def detect_objects(bgr):
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out  = []
    seen = set()

    ranges = [
        ("Red",    (0,   70, 60),  (10,  255, 255)),
        ("Red",    (160, 70, 60),  (180, 255, 255)),
        ("Orange", (10,  70, 60),  (25,  255, 255)),
        ("Yellow", (25,  70, 60),  (35,  255, 255)),
        ("Green",  (36,  55, 55),  (85,  255, 255)),
        ("Blue",   (90,  55, 55),  (128, 255, 255)),
        ("Purple", (129, 50, 50),  (158, 255, 255)),
        ("White",  (0,   0,  210), (180, 22,  255)),
    ]

    for color, lower, upper in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((9,9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 3500: continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            key = (x//70, y//70)
            if key in seen: continue
            seen.add(key)

            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04*peri, True)
            sides  = len(approx)
            ar     = bw / bh if bh > 0 else 1

            if sides == 3:    shape = "Triangle"
            elif sides == 4:  shape = "Square" if 0.85<ar<1.15 else "Rectangle"
            elif sides > 8:   shape = "Circle"
            else:             shape = "Oval"

            area_pct = (area / (w * h)) * 100
            # Minimum 1.5% coverage to avoid tiny false positives
            if area_pct < 1.5: continue

            name, conf, desc = identify_object(color, shape, ar, area_pct, bh, h)
            out.append({
                "label": name, "color": color, "shape": shape,
                "description": desc, "confidence": conf,
                "coverage": round(area_pct, 1),
                "box": [int(x), int(y), int(bw), int(bh)],
            })

    out.sort(key=lambda o: o["coverage"], reverse=True)
    return out[:5]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("index.html", username=session["user"])

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/api/login", methods=["POST"])
def do_login():
    data = request.get_json()
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")
    if username in USERS and USERS[username] == password:
        session["user"] = username
        return jsonify({"success": True, "username": username})
    return jsonify({"success": False, "error": "Invalid username or password"}), 401

@app.route("/api/logout", methods=["POST"])
def do_logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze():
    data  = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image"}), 400
    modes = data.get("modes", ["face","hands","pose","objects"])
    try:
        t0       = time.perf_counter()
        rgb, bgr = decode_image(data["image"])
        out      = {}
        if "face"    in modes: out["faces"]   = detect_faces(bgr)
        if "hands"   in modes: out["hands"]   = detect_hands(bgr)
        if "pose"    in modes: out["pose"]     = detect_pose(bgr)
        if "objects" in modes: out["objects"]  = detect_objects(bgr)
        out["inference_ms"] = int((time.perf_counter() - t0) * 1000)
        out["frame_size"]   = [rgb.shape[1], rgb.shape[0]]
        return jsonify(out)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/status")
def status():
    return jsonify({"status":"ok","user": session.get("user")})

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  VisionPro — Professional Vision AI")
    print("="*55)
    print("  Open: http://127.0.0.1:5000")
    print("  Login: admin / admin123")
    print("  Ctrl+C to stop")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
