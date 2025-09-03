import os, io, uuid, tempfile
from typing import List, Tuple
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import tensorflow as tf

# --------- CONFIG ----------
MODEL_PATH = "model.keras"
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

MAX_SECONDS = 15.0                 # 最多分析 15 秒
DEFAULT_STEP_SEC = 0.5             # 每 0.5 秒抽一幀
THRESHOLD = 0.5                    # 二分類 threshold
# ----------------------------

app = FastAPI(title="Cat Emotion (MVP) API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# static & upload.html
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def index():
    return FileResponse("static/upload.html")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 測試階段開放全部，正式上線更改 domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# iframe header (optional)
@app.middleware("http")
async def add_frame_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Frame-Options"] = "ALLOWALL"
    return resp

# ---- load model ----
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Put your model.keras in project dir.")

model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded model:", MODEL_PATH)
model.summary()

# inspect input shape
try:
    input_shape = model.input_shape  # (None, H, W, C)
except Exception:
    input_shape = (None, 224, 224, 3)
_, IN_H, IN_W, IN_C = input_shape if len(input_shape) == 4 else (None, 224, 224, 3)

# detect whether model already contains a preprocessing/rescaling layer
def model_has_preprocessing(m):
    names = [layer.__class__.__name__.lower() for layer in m.layers]
    # common preprocessing indicators
    for key in ("rescaling", "normalization", "preprocessing", "rescalelayer"):
        if any(key in n for n in names):
            return True
    return False

HAS_INTERNAL_PREPROCESS = model_has_preprocessing(model)
print("Model has internal preprocessing layer:", HAS_INTERNAL_PREPROCESS)

# optionally import preprocess_input for common backbones (used only if model lacks internal preprocess)
use_mobilenet_preprocess = False
if not HAS_INTERNAL_PREPROCESS:
    try:
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
        use_mobilenet_preprocess = True
        print("Will use MobileNetV3 preprocess_input before inference.")
    except Exception:
        use_mobilenet_preprocess = False
        print("No mobilenet preprocess available; will scale images /255 if needed.")

# helper fonts (best-effort)
def load_font(size=28):
    try:
        # try common font locations
        for p in [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ]:
            if os.path.exists(p):
                return ImageFont.truetype(p, size=size)
    except Exception:
        pass
    from PIL import ImageFont
    return ImageFont.load_default()

# -------- preprocessing helpers ----------
def preprocess_frame_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    # Convert BGR -> RGB, resize, and apply preprocessing consistent with training
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
    arr = img.astype("float32")
    if HAS_INTERNAL_PREPROCESS:
        # model expects raw 0-255 inputs
        return arr
    else:
        if use_mobilenet_preprocess:
            return mobilenet_preprocess(arr)
        else:
            # default: scale to [0,1]
            return arr / 255.0

def batch_predict(frames_bgr: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
    if len(frames_bgr) == 0:
        return np.array([])
    arrs = np.stack([preprocess_frame_bgr(f) for f in frames_bgr], axis=0)
    probs = model.predict(arrs, batch_size=batch_size, verbose=0)
    # handle outputs: sigmoid (N,1) or softmax (N,2) or (N,)
    probs = np.array(probs)
    if probs.ndim == 2 and probs.shape[1] == 1:
        return probs.ravel()
    if probs.ndim == 2 and probs.shape[1] == 2:
        # return probability of class index 1
        return probs[:, 1]
    if probs.ndim == 1:
        return probs.ravel()
    # fallback: take last dim as class1 probability if >1 dims
    return probs.reshape(len(frames_bgr), -1)[:, -1]

# -------- frame extraction ----------
def extract_frames(video_path: str, step_sec: float = DEFAULT_STEP_SEC, max_seconds: float = MAX_SECONDS) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0
    use_seconds = min(duration, max_seconds)
    step_frames = max(int(round(fps * step_sec)), 1)
    frames = []
    frame_idx = 0
    read_idx = 0
    total_to_read = int(round(fps * use_seconds))
    # iterate frames; sample every step_frames
    while read_idx < total_to_read:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step_frames == 0:
            frames.append(frame)
        frame_idx += 1
        read_idx += 1
    cap.release()
    return frames, use_seconds

# -------- aggregation & representative frame ----------
def aggregate_probs(probs: np.ndarray, threshold: float = THRESHOLD):
    preds = (probs >= threshold).astype(int)
    n = len(preds)
    if n == 0:
        return 0, 0.0, 0.0
    count1 = int(preds.sum())
    count0 = n - count1
    ratio1 = count1 / n
    ratio0 = count0 / n
    label = 1 if count1 >= count0 else 0
    return int(label), float(ratio0), float(ratio1)

def pick_representative(frames: List[np.ndarray], probs: np.ndarray, label: int):
    if len(frames) == 0:
        return None
    if label == 1:
        idx = int(np.argmax(probs))
    else:
        idx = int(np.argmin(probs))
    idx = max(0, min(idx, len(frames)-1))
    return frames[idx]

# -------- card drawing ----------
def make_card(frame_bgr, final_label:int, ratio0:float, ratio1:float, seconds:float) -> str:
    W, H = 960, 540
    # colors
    relaxed_bg = (235,248,242)
    uncomfy_bg = (255,240,232)
    bg = uncomfy_bg if final_label==1 else relaxed_bg
    card = Image.new("RGB", (W,H), bg)
    draw = ImageDraw.Draw(card)
    font_t = load_font(40)
    font_b = load_font(24)
    # left thumbnail
    if frame_bgr is not None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        thumb_h = H - 80
        thumb_w = int(thumb_h * 16 / 9)
        thumb = Image.fromarray(cv2.resize(rgb, (thumb_w, thumb_h)))
        card.paste(thumb, (40,40))
    # right texts
    x0 = 40 + thumb_w + 30
    y = 60
    title = "不適" if final_label==1 else "放鬆"
    draw.text((x0, y), f"結果：{title}", fill=(10,10,10), font=font_t)
    y += 60
    draw.text((x0, y), f"影片時長：約{seconds:.1f}秒", fill=(30,30,30), font=font_b)
    y += 40
    draw.text((x0, y), f"放鬆比例：{ratio0*100:.1f}%", fill=(30,30,30), font=font_b)
    y += 30
    draw.text((x0, y), f"不適比例：{ratio1*100:.1f}%", fill=(30,30,30), font=font_b)
    y += 40
    desc = "主子耳朵下壓或緊繃，可能處於不適或壓力中，請多多留意主子的狀況！" if final_label==1 else "主子耳朵自然、表情放鬆，狀態穩定。可以適時嚕主子！"
    draw.text((x0, y), desc, fill=(30,30,30), font=font_b)
    fname = f"card_{uuid.uuid4().hex}.jpg"
    path = os.path.join(STATIC_DIR, fname)
    card.save(path, quality=90)
    return fname

# -------- APIs ----------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/analyze")
async def analyze_test(file: UploadFile = File(...),
                       step_sec: float = Query(DEFAULT_STEP_SEC, ge=0.1, le=2.0),
                       threshold: float = Query(THRESHOLD, ge=0.05, le=0.95),
                       clip15s: bool = Query(True)):
    """
    上傳影片（form-data key=file），回傳 JSON 與圖卡 URL。
    """
    # save temp
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            data = await file.read()
            tmp.write(data)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot save uploaded file: {e}")

    try:
        frames, used_seconds = extract_frames(tmp_path, step_sec=step_sec, max_seconds=MAX_SECONDS if clip15s else MAX_SECONDS)
        if not frames:
            raise HTTPException(status_code=400, detail="No frames extracted from video.")
        probs = batch_predict(frames)
        final_label, ratio0, ratio1 = aggregate_probs(probs, threshold=threshold)
        rep = pick_representative(frames, probs, final_label)
        card_fname = make_card(rep, final_label, ratio0, ratio1, used_seconds)
        # cleanup temp
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return {
            "filename": file.filename,
            "frames_used": len(frames),
            "seconds_analyzed": round(used_seconds,2),
            "result": "不適" if final_label==1 else "放鬆",
            "ratios": {"relaxed": round(ratio0,4), "uncomfortable": round(ratio1,4)},
            "card_url": f"/static/{card_fname}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- run (for dev) ----------
# uvicorn main:app --host 0.0.0.0 --port 8000