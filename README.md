VisionPro AI
Real-time Vision Detection Web Application
1)Python
2)Flask 
3)OpenCV  
4)Real-time 
5)Web App
It contains Face & Emotion, Hand Gestures, Body Pose, Object Detection, Auth + Login.

Project Overview:
VisionPro is a professional real-time AI vision detection web application. It uses your webcam to detect and analyze facial emotions, hand gestures, body pose actions, and colored objects — all live in the browser through a modern, secure web interface.
The application requires no machine learning training — it uses pre-built OpenCV models that work immediately after installation.

Key Features:
---Face & Emotion Detection
* Detects faces in real-time using OpenCV Haar Cascade classifier
* Recognizes 9 emotions: Happy, Laughing, Sad, Crying, Angry, Surprised, Fearful, Disgusted, Thinking, Neutral
* Counts visible eyes per face
* Displays emotion confidence percentage

---Hand Gesture Recognition
* Detects hands using skin color segmentation (HSV color space)
* Counts fingers using convex hull + convexity defect analysis
* Recognizes: Fist, Open Hand, Peace, Pointing, Thumbs Up, OK Sign, Three/Four Fingers
*	Strict false-positive filtering — avoids detecting fingers when hand is not shown

---Body Pose & Actions
* Detects full body and upper body using OpenCV cascade classifiers
*	Infers actions: Standing, Sitting, Crouching, Left/Right of Frame, Close/Far from Camera, Jumping

---Object Identification
* Detects colored objects: Red, Orange, Yellow, Green, Blue, Purple, White
* Identifies real objects: Apple, Orange, Banana, Ball, Bottle, Book, Phone, Laptop, Paper, Cup, Box and more
* Uses color + shape + aspect ratio + size for accurate identification

---Authentication & UI
*Login / Logout system with session management
* 3 demo accounts included (admin, demo, user)
* Day / Night theme toggle (saved in browser)
* Professional dashboard with detection cards, confidence bars, history tab
* Real-time FPS counter and inference time display
  
---Technology Stack
1) Component	Technology
2) Backend	Python 3.x + Flask
3) Vision AI	OpenCV (cv2) — Haar Cascades + Color Segmentation
4) Image Processing	NumPy + Pillow (PIL)
5) Frontend	HTML5 + CSS3 + Vanilla JavaScript
6) Camera	WebRTC (getUserMedia API)
7) Canvas	HTML5 Canvas — real-time bounding box overlay
8) Auth	Flask Session (server-side)
9) Fonts	Google Fonts — Playfair Display + DM Sans

Project Structure
visionpro/
   app.py                  ← Main Flask application
   requirements.txt        ← Python dependencies
   RUN.bat                 ← One-click Windows launcher
   templates/
      login.html          ← Professional login page
      index.html          ← Main dashboard
static/
     css/                ← (optional) extra stylesheets
     js/                 ← (optional) extra scripts

---Installation & Setup
Prerequisites
•	Python 3.8 or higher
•	pip (Python package manager)
•	Google Chrome or Microsoft Edge browser
•	Webcam connected to your computer

Step 1 — Download & Extract
Download visionpro.zip and extract it to a folder on your computer.
Extracted folder: C:\Users\admin\visionpro

Step 2 — Install Packages
Open Command Prompt (Press Windows + R, type cmd, press Enter):
python -m pip install flask flask-cors numpy Pillow opencv-python
This only needs to be done once. Wait for all packages to install.

Step 3 — Run the Application
cd C:\Users\admin\visionpro
python app.py
You will see:
  VisionPro — Professional Vision AI
  Open: http://127.0.0.1:5000
  Login: admin / admin123

Step 4 — Open Browser
Open Chrome or Edge and go to:
http://127.0.0.1:5000

Step 5 — Login
Username	Password	Role
admin	admin123	Full Access
demo	demo123	Demo User
user	password	Basic User

Step 6 — Start Detection
Click Start Detection button — allow camera access in browser — detection begins automatically.

---Running the App (Quick Reference)
Every time you want to run VisionPro:
cd C:\Users\admin\visionpro
python app.py
Then open http://127.0.0.1:5000 in Chrome.
To stop: press Ctrl + C in the Command Prompt window.
Or double-click RUN.bat to install packages and run in one step.

---What to Show the Camera
What You Do	What It Detects
Look at camera and smile Happy
Open mouth wide Surprised
Frown your eyebrows	 Angry
Make a fist	Fist
Show peace sign  Peace Sign
Show thumbs up	 Thumbs Up
Open all five fingers	 Open Hand
Raise your arm up	 Arm Raised
Sit down in front of camera	 Sitting
Hold a red apple	Apple
Hold a blue bottle	 Blue Bottle
Hold a green bottle	 Green Bottle
Show a yellow object	 Lemon / Ball

---How It Works
The browser captures a frame from the webcam every 700ms using the WebRTC API and HTML5 Canvas. The frame is encoded as a base64 JPEG and sent to the Flask server via a POST request to /api/analyze.
The server processes the frame using OpenCV:
•	Face detection — Haar Cascade classifier on grayscale frame
•	Emotion analysis — 5-zone brightness analysis on face ROI (Region of Interest)
•	Hand detection — HSV skin color mask + contour + convexity defects
•	Gesture classification — Finger counting from convex hull defect angles
•	Body detection — Full body + upper body Haar Cascade
•	Object detection — HSV color range masking + shape analysis + size/aspect ratio
Results are returned as JSON. The browser draws bounding boxes on a Canvas overlay on top of the video feed and updates the sidebar in real time.

---Detection Tips for Best Accuracy
Face & Emotion
•	Good lighting on your face — avoid backlighting
•	Face the camera directly at 0.5–1.5 metres distance
•	Exaggerate expressions for better detection (open mouth wide for surprised/angry)

Hand Gestures
•	Show your hand clearly against a plain background
•	Hold gestures steady for 1–2 seconds
•	Avoid wearing gloves or having complex backgrounds
•	Keep hand at least 30cm from camera (not too close)

Objects
•	Use clearly colored objects (bright red, blue, green, yellow, orange)
•	Object should occupy at least 2% of the camera frame
•	Plain or simple backgrounds work best

---Troubleshooting
Problem	Solution
Website not opening	Make sure cmd is open and python app.py is running. Use http:// not https://
Camera not starting	Allow camera permission in browser. Try Chrome instead of Edge.
ModuleNotFoundError	Run: python -m pip install flask flask-cors numpy Pillow opencv-python
Port already in use	Run: netstat -ano | findstr :5000  then taskkill /PID <number> /F
Login not working	Use admin / admin123 exactly. Check for extra spaces.
False hand gestures	Ensure good lighting. Keep background simple. Hold gesture for 1-2 seconds.
Wrong object detected	Use brightly colored objects. Plain background helps. Object must be large enough.
TensorFlow warnings in cmd	These are harmless info messages from OpenCV. Ignore them.

🔌  API Reference
Endpoint	Method   Description
GET  /	Main dashboard (requires login)
GET  /login	Login page
POST /api/login	Authenticate user → returns {success, username}
POST /api/logout	Clear session → redirect to login
POST /api/analyze	Analyze image frame → returns detections JSON
GET  /api/status	Server status check

POST /api/analyze — Request Body
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "modes": ["face", "hands", "pose", "objects"]
}

POST /api/analyze — Response
{
  "faces":   [{ "box": [x,y,w,h], "emotion": "Happy", "confidence": 87 }],
  "hands":   [{ "gesture": " Peace Sign", "meaning": "Two fingers raised" }],
  "pose":    { "detected": true, "actions": [{"action": " Standing", "detail": "..."}] },
  "objects": [{ "label": "Apple", "color": "Red", "confidence": 87 }],
  "inference_ms": 43
}

---Requirements
flask>=3.0.0
flask-cors>=4.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0

---Built With
•	Python + Flask — Web framework and REST API
•	OpenCV (cv2) — Computer vision and image processing
•	NumPy — Numerical array processing for image data
•	Pillow — Image decoding and format conversion
•	WebRTC — Browser-based camera access
•	HTML5 Canvas — Real-time bounding box drawing over video

VisionPro AI  •  Built with Python & OpenCV  •  Real-time Vision Detection
