# streamlit_app.py
import os, io, json, math, uuid, pickle, base64, random
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import cv2
import pandas as pd
import streamlit as st
from textwrap import dedent 
from streamlit.components.v1 import html as st_html
from streamlit.components.v1 import html  # tambahkan sekali di atas

# ==== TF/Keras ====
import tensorflow as tf
import keras
from keras import layers, Model, Input
from keras.applications import EfficientNetB0  # hanya jika nanti rebuild model
from keras.applications.efficientnet import preprocess_input as effnet_preprocess

# ==== fitur manual ====
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import mimetypes, base64

def to_data_uri(p: Path) -> str:
    if not p.exists():
        return ""
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64  = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"

# ------------------ konfigurasi umum ------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
try:
    keras.utils.set_random_seed(SEED)
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EXPORT_DIR = STATIC_DIR / "exports"
UPLOAD_DIR = STATIC_DIR / "uploads"
for d in [STATIC_DIR, EXPORT_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# default, bisa di-override oleh meta.json
IMG_SIZE: Tuple[int, int] = (224, 224)
USE_MASKED_INPUT = True         # putihkan background biar tahan domain shift
MIN_AREA_RATIO   = 0.005        # 0.5% area kontur minimal

# --- lokasi artefak model ---
ARTIFACT_DIR = BASE_DIR / "artifactsv2"
CANDIDATE_MODELS = [
    ARTIFACT_DIR / "best_hybrid.keras",      # hybrid (.keras)
    ARTIFACT_DIR / "best_hybrid.h5",         # hybrid (h5)
    BASE_DIR / "best_model_compressed.h5",   # CNN-only (file kamu)
]

KEY_FEATURES = [
    "area","perimeter","form_factor","aspect_ratio","extent","solidity","eccentricity",
    "glcm_contrast","glcm_energy","glcm_homogeneity","h_mean","s_mean","v_mean"
]

# ------------------ util UI ------------------
st.set_page_config(page_title="Deteksi Daun (Streamlit)", layout="wide")
st.sidebar.title("⚙️ Pengaturan")

# ------------------ loader fleksibel ------------------
@st.cache_resource(show_spinner=True)
def load_model_flex() -> keras.Model:
    last_err = None
    for p in CANDIDATE_MODELS:
        if p.exists():
            try:
                return keras.models.load_model(p)
            except Exception as e:
                last_err = e
    raise RuntimeError(
        f"Tidak menemukan model yang valid. Dicoba: {CANDIDATE_MODELS}. Error terakhir: {last_err}"
    )

@st.cache_resource(show_spinner=False)
def load_labeling_and_scaler():
    meta = scaler = le = None
    meta_p   = ARTIFACT_DIR / "meta.json"
    scaler_p = ARTIFACT_DIR / "scaler.pkl"
    le_p     = ARTIFACT_DIR / "label_encoder.pkl"
    if meta_p.exists() and scaler_p.exists() and le_p.exists():
        with open(meta_p) as f: meta = json.load(f)
        with open(scaler_p, "rb") as f: scaler = pickle.load(f)
        with open(le_p, "rb") as f: le = pickle.load(f)
    return meta, scaler, le

model = load_model_flex()
meta, scaler, le = load_labeling_and_scaler()

try:
    n_inputs = len(model.inputs)
except Exception:
    n_inputs = 1

IS_HYBRID = (meta is not None and scaler is not None and le is not None and n_inputs == 2)
FEATURE_COLS: List[str] = (meta.get("feature_cols", []) if meta else [])
CLASSES: List[str] = (meta.get("classes", []) if meta else [])

if not IS_HYBRID and not CLASSES:
    try:
        n = int(model.outputs[0].shape[-1])
        CLASSES = [f"class_{i}" for i in range(n)]
    except Exception:
        CLASSES = []

if meta and "img_size" in meta:
    IMG_SIZE = tuple(meta["img_size"])

# ------------------ fitur manual (dipakai bila hybrid) ------------------
def segment_leaf_mask(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return np.zeros((1,1), np.uint8)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    _, th_s = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_v = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(th_s, th_v)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return mask
    c = cnts[int(np.argmax([cv2.contourArea(x) for x in cnts]))]
    out = np.zeros((h,w), dtype=np.uint8)
    cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
    if cv2.contourArea(c) < MIN_AREA_RATIO * (h*w):
        out = mask
    return out

def shape_features(mask: np.ndarray) -> Dict[str, float]:
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return {k: 0.0 for k in [
            "area","perimeter","form_factor","aspect_ratio","extent",
            "solidity","eccentricity", *[f"hu_{i+1}" for i in range(7)]
        ]}
    c = cnts[int(np.argmax([cv2.contourArea(x) for x in cnts]))]
    area = cv2.contourArea(c)
    per  = cv2.arcLength(c, True)
    ff   = 4*np.pi*area/(per**2 + 1e-8)
    x,y,bw,bh = cv2.boundingRect(c)
    ar   = bw/(bh + 1e-8)
    extent = area/(bw*bh + 1e-8)
    hull = cv2.convexHull(c)
    ha   = cv2.contourArea(hull)
    solidity = area/(ha + 1e-8) if ha>0 else 0.0
    if len(c)>=5:
        (_, _),(MA,ma),_ = cv2.fitEllipse(c)
        a = max(MA,ma)/2.0; b = min(MA,ma)/2.0
        ecc = np.sqrt(max(a*a - b*b, 0.0))/(a + 1e-8) if a>0 else 0.0
    else:
        ecc = 0.0
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    hu_log = [(-1 if h<0 else 1)*np.log10(abs(h)+1e-30) for h in hu]
    out = {
        "area": float(area/(h*w)), "perimeter": float(per/(h+w)),
        "form_factor": float(ff), "aspect_ratio": float(ar),
        "extent": float(extent), "solidity": float(solidity), "eccentricity": float(ecc),
    }
    for i,v in enumerate(hu_log):
        out[f"hu_{i+1}"] = float(v)
    return out

def texture_features(gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    gm = np.where(mask>0, gray, 0)
    gq = (gm/32).astype(np.uint8)
    glcm = graycomatrix(
        gq, distances=[1,2,4],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=8, symmetric=True, normed=True
    )
    props = {}
    for prop in ["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]:
        props[prop] = float(np.mean(graycoprops(glcm, prop)))
    lbp = local_binary_pattern(gm, P=8, R=1, method="uniform")
    lbp_masked = lbp[mask>0]
    n_bins = 8+2
    hist, _ = np.histogram(lbp_masked, bins=np.arange(0, n_bins+1), density=True)
    out = {f"glcm_{k}": v for k,v in props.items()}
    out.update({f"lbp_{i}": float(hist[i]) for i in range(n_bins)})
    return out

def color_features(bgr: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    valid = mask>0
    if np.count_nonzero(valid)==0:
        return {"h_mean":0.0,"s_mean":0.0,"v_mean":0.0,"h_std":0.0,"s_std":0.0,"v_std":0.0}
    H = hsv[:,:,0][valid]; S = hsv[:,:,1][valid]; V = hsv[:,:,2][valid]
    return {
        "h_mean": float(np.mean(H)), "s_mean": float(np.mean(S)), "v_mean": float(np.mean(V)),
        "h_std": float(np.std(H)),  "s_std": float(np.std(S)),  "v_std": float(np.std(V)),
    }

def extract_manual_features_from_bgr(bgr: np.ndarray) -> Dict[str, float]:
    mask = segment_leaf_mask(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    feats = {}
    feats.update(shape_features(mask))
    feats.update(texture_features(gray, mask))
    feats.update(color_features(bgr, mask))
    return feats

# ------------------ preprocess gambar ------------------
def mask_background_to_white(bgr: np.ndarray) -> np.ndarray:
    mask = segment_leaf_mask(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[mask == 0] = 255
    return rgb

def preprocess_for_model(bgr: np.ndarray) -> np.ndarray:
    if USE_MASKED_INPUT:
        rgb = mask_background_to_white(bgr)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, IMG_SIZE)
    img = img.astype(np.float32)
    img = effnet_preprocess(img)
    return img

def annotated_copy(bgr: np.ndarray, label: str, conf: float, color=(0, 0, 0)) -> np.ndarray:
    out = bgr.copy()
    txt = f"{label} ({conf:.2f})"
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.40, min(0.90, (min(h, w) / 800.0) * 0.6))
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(txt, font, scale, thickness)
    x, y = 10, 10 + th
    cv2.putText(out, txt, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return out

def annotated_with_features(
    bgr: np.ndarray,
    pred: str,
    conf: float,
    feats: Dict[str, float],
    draw_label_on_image: bool = True
) -> np.ndarray:
    vis = bgr.copy()
    mask = segment_leaf_mask(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return vis
    c = cnts[int(np.argmax([cv2.contourArea(x) for x in cnts]))]

    cv2.drawContours(vis, [c], -1, (0,255,0), 2)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(vis, (x,y), (x+w, y+h), (255,255,0), 2)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(vis, [box], True, (0,165,255), 2)

    if len(c) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(c)
        cv2.ellipse(vis, ((int(cx), int(cy)), (int(MA), int(ma)), angle), (255,0,255), 2)
        a = MA/2.0; b = ma/2.0
        ang = math.radians(angle)
        dx, dy = math.cos(ang), math.sin(ang)
        p1 = (int(cx - a*dx), int(cy - a*dy))
        p2 = (int(cx + a*dx), int(cy + a*dy))
        cv2.line(vis, p1, p2, (0,0,255), 2)
        pdx, pdy = -dy, dx
        p3 = (int(cx - b*pdx), int(cy - b*pdy))
        p4 = (int(cx + b*pdx), int(cy + b*pdy))
        cv2.line(vis, p3, p4, (255,0,0), 2)

    M = cv2.moments(c)
    if M["m00"] != 0:
        ccx, ccy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        cv2.circle(vis, (ccx, ccy), 4, (0,255,255), -1)

    if draw_label_on_image:
        vis = annotated_copy(vis, pred, conf, color=(0, 0, 0))
    return vis

# ------------------ prediksi fleksibel ------------------
def predict_bgr(bgr: np.ndarray):
    X_img = np.expand_dims(preprocess_for_model(bgr), axis=0)
    if IS_HYBRID:
        feats = extract_manual_features_from_bgr(bgr)
        vec = np.array([[feats.get(c, 0.0) for c in FEATURE_COLS]], dtype=np.float32)
        vec = scaler.transform(vec)
        probs = model.predict([X_img, vec], verbose=0)[0]
    else:
        probs = model.predict(X_img, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    pred_label = (CLASSES[top_idx] if CLASSES else f"class_{top_idx}")
    order = np.argsort(probs)[::-1][:3]
    topk = [(CLASSES[i] if CLASSES else f"class_{i}", float(probs[i])) for i in order]
    return pred_label, top_conf, topk

# --- mapping label → slug halaman species ---
from typing import Optional

def label_to_slug(lbl: str) -> Optional[str]:
    """Terima label model (kode singkat atau nama lengkap, case-insensitive),
    kembalikan slug untuk router ?page=species&sp=..."""
    if not lbl:
        return None
    t = lbl.lower().strip()
    t = t.replace("-", " ").replace("_", " ").replace(".", " ")
    t = " ".join(t.split())  # normalisasi spasi
    MAP = {
        # Avicennia marina
        "am": "avicennia_marina",
        "avicennia marina": "avicennia_marina",
        "avicennia_marina": "avicennia_marina",
        # Avicennia officinalis
        "ao": "avicennia_officinalis",
        "avicennia officinalis": "avicennia_officinalis",
        "avicennia_officinalis": "avicennia_officinalis",
        # Bruguiera gymnorhiza
        "bg": "bruguiera_gymnorhiza",
        "bruguiera gymnorhiza": "bruguiera_gymnorhiza",
        "bruguiera_gymnorhiza": "bruguiera_gymnorhiza",
        # Heritiera littoralis
        "hl": "heritiera_littoralis",
        "heritiera littoralis": "heritiera_littoralis",
        "heritiera_littoralis": "heritiera_littoralis",
        # Lumnitzera littorea
        "lt": "lumnitzera_littorea",
        "lumnitzera littorea": "lumnitzera_littorea",
        "lumnitzera_littorea": "lumnitzera_littorea",
        # Rhizophora apiculata
        "ra": "rhizophora_apiculata",
        "rhizophora apiculata": "rhizophora_apiculata",
        "rhizophora_apiculata": "rhizophora_apiculata",
        # Sonneratia alba
        "sa": "sonneratia_alba",
        "sonneratia alba": "sonneratia_alba",
        "sonneratia_alba": "sonneratia_alba",
    }
    return MAP.get(t)
  
# ------------------ Streamlit UI ------------------
# CSS mirip style di template
CUSTOM_CSS = """
<style>
:root{
  --green:#2C4001; --green-2:#365902;
}
.nav{
  position: sticky;
  top:0;
  z-index:100;
  background: var(--green);
  color:#fff;
  width:100%;
  margin:0 0 18px 0;
  box-shadow: 0 2px 6px rgba(0,0,0,.15);
}
.nav-inner{
  max-width:1200px;
  margin:0 auto;
  padding:10px 16px;
  display:flex;
  align-items:center;
  justify-content:center;  /* menu jadi di tengah */
}
/* hapus brand (biar MangroveLeaf hilang) */
.brand{ display:none; }

.nav-inner nav{
  display:flex;
  gap:24px;
}
.nav-inner nav a{
  color:#f2f2f2;
  text-decoration:none;
  padding:8px 16px;
  border-radius:8px;
  font-weight:600;
}
.nav-inner nav a.active,
.nav-inner nav a:hover{
  background:rgba(255,255,255,.12);
  color:#fff;
}
/* HERO */
.hero{ max-width:1200px; margin:0 auto 16px auto; background:#fff; border-radius:16px; padding:28px; box-shadow: var(--shadow); }
.hero h1{ font-size:42px; line-height:1.15; margin:0 0 10px 0; color:#1d2a1a; }
.hero p.lead{ font-size:16px; color:#445; margin:6px 0 18px 0; }
.hero .btns{ display:flex; gap:12px; flex-wrap:wrap; }
.hero-grid{ display:grid; grid-template-columns:1.25fr 1fr; gap:24px; align-items:center; }
.hero img{ border-radius:12px; display:block; width:100%; height:auto; }
.btn-primary{ background: var(--green); color:#fff !important; padding:12px 16px; border-radius:10px; text-decoration:none; display:inline-block; font-weight:700; border:none; }
.btn-secondary{ background:#eef5ee; color:#1d2a1a !important; padding:12px 16px; border-radius:10px; text-decoration:none; display:inline-block; font-weight:700; border:1px solid #dde7dd; }
.btn-primary:hover{ filter:brightness(1.05); }
.btn-secondary:hover{ background:#e7efe7; }

/* SECTION TITLE */
.section{ max-width:1200px; margin:22px auto; }
.section h3{ font-size:26px; margin:0 0 6px 0; color:#2C4001; text-align:center; font-weight:800; }
.section p{ color:#3C4030; text-align:center; margin:0 0 16px 0; }

/* NEWS CARDS */
.news-scroller{
  display:flex;
  flex-wrap:nowrap;
  gap:20px;
  overflow-x:auto;
  padding:10px;
  scroll-snap-type:x mandatory;
  -webkit-overflow-scrolling:touch;
}
.news-scroller::-webkit-scrollbar{ height:10px; }
.news-scroller::-webkit-scrollbar-thumb{ background:#cfd6c6; border-radius:6px; }

/* Kartu berita fix ukuran */
.news-item{ flex:0 0 280px; scroll-snap-align:start; }  /* lebar fix 280px */
.card-news{
  background:#fff;
  border-radius:12px;
  overflow:hidden;
  box-shadow:0 6px 20px rgba(0,0,0,.08);
  border:1px solid #e6eadf;
  display:flex;
  flex-direction:column;
  height:100%;
}
.card-news img{
  width:100%;
  height:180px;          /* tinggi gambar fix */
  object-fit:cover;
  display:block;
}
.card-body{ padding:14px 16px 16px 16px; }
.badge{ display:inline-block; background:var(--green-2); color:#fff;
  padding:4px 10px; font-size:12px; border-radius:999px; font-weight:700; }
.card-title{ margin:8px 0 6px 0; font-weight:800; font-size:18px; color:#254018; line-height:1.3; }
.card-snippet{ font-size:14px; color:#4a4f45; line-height:1.5; min-height:48px; margin-bottom:8px; }
.card-footer{ display:flex; justify-content:space-between; align-items:center;
  font-size:13px; color:#667; }
.card-footer .cta{ color:var(--green-2); font-weight:700; text-decoration:none; }
.card-footer .cta:hover{ text-decoration:underline; }

/* GREEN RAIL UNDER SCROLLER */
.news-rail{ height:6px; max-width:1200px; margin:8px auto 0 auto;
  background:linear-gradient(90deg,#2C4001 0%,#7fa15b 100%); border-radius:3px; position:relative; }
.news-rail:before,.news-rail:after{
  content:""; position:absolute; top:-3px; width:0; height:0; border-top:6px solid transparent; border-bottom:6px solid transparent;
}
.news-rail:before{ left:-6px; border-right:6px solid #2C4001; }
.news-rail:after{ right:-6px; border-left:6px solid #7fa15b; }

/* Footer */
.footer{ max-width:1200px; margin:26px auto 18px auto; text-align:center; color:#678; font-size:12px; border-top:1px solid #eef0ee; padding-top:12px; }

/* Responsif */
@media (max-width:600px){
  .news-item{ flex:0 0 85%; }
  .card-news img{ height:160px; }
}

/* ===== ABOUT PAGE ===== */
.about-wrap{ max-width:1200px; margin:0 auto; padding:18px 14px 28px; }
.about-title{ text-align:center; color:var(--green); font-weight:900; margin:6px 0 18px; font-size:28px; }

/* Ekosistem Mangrove */
.ekos-grid{ display:grid; grid-template-columns:1fr 1fr; gap:28px; align-items:center; margin-bottom:28px; }
.ekos-img{ position:relative; }
.ekos-img img{ width:100%; border-radius:12px; border:2px solid #434736; box-shadow:0 10px 26px rgba(0,0,0,.15); display:block; }
.ekos-badge{ position:absolute; left:12px; bottom:12px; background:#365902; color:#fff; border-radius:8px; padding:8px 10px; font-size:12px; font-weight:700; box-shadow:0 3px 8px rgba(0,0,0,.2); }
.ekos-card{ background:#fff; border-left:4px solid var(--green-2); border-radius:10px; padding:16px 18px; box-shadow:0 8px 20px rgba(0,0,0,.08); }
.ekos-card h3{ color:var(--green); margin:0 0 8px 0; }
.ekos-card ul{ margin:8px 0 0 18px; color:#3C4030; }

/* Ciri Khas Daun (kotak kiri + gambar kanan) */
.leaf-grid{ display:grid; grid-template-columns:1fr 1fr; gap:28px; align-items:center; margin:6px 0 30px; }
.leaf-card{ background:#fff; border:1px solid #ccd2c9; border-radius:14px; box-shadow:0 8px 20px rgba(0,0,0,.08); overflow:hidden; }
.leaf-card .sec{ padding:18px 20px; border-bottom:1px solid #eef0ea; }
.leaf-card .sec:last-child{ border-bottom:none; }
.leaf-card h5{ margin:0 0 8px 0; color:#365902; font-size:22px; }
.leaf-card ul{ margin:0 0 0 18px; color:#3C4030; }
.leaf-img{ width:100%; border-radius:12px; border:2px solid #434736; box-shadow:0 10px 24px rgba(0,0,0,.15); display:block; }

/* Subjudul kecil */
.subtle{ text-align:center; color:#434736; margin:-6px 0 14px; font-size:14px; }

/* ===== Carousel Jenis-Jenis (7 item satu baris, seperti gambar) ===== */
.carousel{ position:relative; max-width:1240px; margin:0 auto 6px; }

.track{
  display:flex;
  flex-wrap:nowrap;                 /* semua item di satu baris */
  gap:32px;                         /* jarak antar kartu */
  padding:0 64px 18px;              /* ruang untuk tombol panah */
  overflow-x:auto;
  scroll-snap-type:x proximity;
  -webkit-overflow-scrolling:touch;
  scrollbar-width:thin; 
  scrollbar-color:#365902 #e7efe5;
}
.track::-webkit-scrollbar{ height:8px; }
.track::-webkit-scrollbar-thumb{ background:#365902; border-radius:10px; }

/* kotak kartu fix ~320px seperti contoh */
.card-spec{
  flex:0 0 320px;                   /* lebar kartu fix (4 kartu muat di 1240px) */
  min-width:320px;
  background:#fff;
  border:1px solid #cfd6c6;
  border-radius:12px;
  overflow:hidden;
  box-shadow:0 6px 18px rgba(0,0,0,.08);
  scroll-snap-align:start;
  transition:.25s transform,.25s box-shadow;
}
.card-spec:hover{ transform:translateY(-4px); box-shadow:0 14px 26px rgba(54,89,2,.20); }

/* gambar konsisten 4:3 */
.card-spec img{
  width:100%;
  aspect-ratio:4/3;
  object-fit:cover;
  display:block;
}
/* fallback jika browser tak dukung aspect-ratio */
@supports not (aspect-ratio: 4 / 3){
  .card-spec img{ height:240px; }
}

.card-spec .body{
  flex: 1 1 auto;                            /* isi mengisi sisa tinggi */
  display: flex; 
  flex-direction: column;
}

.card-spec h5{ color:#2C4001; margin:2px 0 10px; font-size:20px; font-weight:800; padding:0 16px; text-align:center; }

/* Tombol kiri/kanan, posisinya menjorok ke luar seperti gambar */
.cbtn{
  position:absolute; top:50%; transform:translateY(-50%);
  width:44px; height:44px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  background:#365902; color:#fff; border:2px solid #fff;
  box-shadow:0 2px 10px rgba(0,0,0,.15); cursor:pointer; transition:.2s;
}
.cbtn:hover{ background:#2C4001; transform:translateY(-50%) scale(1.07); }
.cbtn.left{ left:-22px; }          /* sedikit keluar sisi kiri */
.cbtn.right{ right:-22px; }        /* sedikit keluar sisi kanan */

/* Rail progres di bawah + titik di ujung seperti contoh */
.rail,
.rail .thumb,
.thumb {
  display: none !important;
  height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
}
/* Responsif: 3/2/1 kartu yang terlihat */
@media (max-width:1199px){ .card-spec{ flex:0 0 calc((100% - 2*32px)/3); min-width:0; } }
@media (max-width:900px){
  .ekos-grid, .leaf-grid{ grid-template-columns:1fr; }
  .card-spec{ flex:0 0 calc((100% - 32px)/2); }
  .cbtn{ width:38px; height:38px; }
}
@media (max-width:600px){
  .card-spec{ flex:0 0 100%; }
  .cbtn{ display:none; }
}

/* ===== FAQ ===== */
.faq-wrap{max-width:800px;margin:0 auto 28px;}
.faq-title{color:#2C4001;text-align:center;margin:14px 0 18px;font-weight:900;}
.accordion{display:block}
.ac-item{border:1px solid #3C4030;border-radius:8px;overflow:hidden;margin-bottom:12px;background:#fff}

/* pakai details/summary agar tanpa JS */
.ac-item details{border-radius:8px;overflow:hidden;background:#fff}
.ac-item summary{
  list-style:none; cursor:pointer; user-select:none;
  background:#365902; color:#fff; padding:12px 16px; font-weight:700;
}
.ac-item summary::-webkit-details-marker{display:none}
.ac-item[open] summary{background:#2C4001}
.ac-body{background:#f8f9fa;color:#3C4030;padding:14px 16px}

/* ikon chevron */
.ac-item summary::after{
  content:""; float:right; width:16px; height:16px;
  mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path fill="%23000" d="M1.646 4.646a.5.5 0 0 1 .708 0L8 10.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/></svg>') no-repeat center / contain;
  background:#fff;
  transition:transform .2s ease;
}
.ac-item[open] summary::after{transform:rotate(180deg)}

/* fokus/hover ala bootstrap */
.ac-item summary:focus{outline:none; box-shadow:0 0 0 0.25rem rgba(54,89,2,.25)}
.ac-item summary:hover{filter:brightness(1.05)}

@media (max-width:768px){ .accordion{padding:0 12px} }

 .sp-list.two-col { 
    columns: 2;           /* Firefox/Chromium */
    -webkit-columns: 2;   /* Safari */
    -moz-columns: 2;
    column-gap: 28px;
    padding-left: 0;
  }
  .sp-list.two-col li { 
    break-inside: avoid;
  }
  @media (max-width: 780px){
    .sp-list.two-col { columns: 1; -webkit-columns: 1; -moz-columns: 1; }
  }

/* ===== Footer sejajar dengan kotak navbar ===== */
.site-footer{
  /* tidak lagi full-bleed; biarkan wrapper transparan */
  background: transparent;
  padding: 0;                /* padding ada di .inner agar match .nav-inner */
  margin: 24px 0 0;          /* jarak kecil dari konten terakhir */
  box-shadow: none;
}
.site-footer .inner{
  max-width: 1200px;         /* sama dengan .nav-inner */
  margin: 0 auto;
  background: var(--green);  /* bar hijau seperti navbar */
  color: #fff;
  text-align: center;
  padding: 12px 16px;        /* sejajar dengan padding .nav-inner */
  box-shadow: 0 -2px 8px rgba(0,0,0,.18);
  font-weight: 600;
  border-radius: 0;          /* ubah ke 10px kalau mau rounded */
  letter-spacing: .2px;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



# “Routing” sederhana via query param ?page=home|deteksi
try:
    page = st.query_params.get("page", "home")
    if isinstance(page, list):  # antisipasi tipe list
        page = page[0] if page else "home"
except Exception:
    page = "home"
if page not in ("home", "deteksi", "about", "faq", "species"):
    page = "home"

def nav_link(caption: str, target: str, active: bool) -> str:
    # pakai URL relatif + target=_self agar tetap di tab yang sama
    href = f"?page={target}"
    cls = "active" if active else ""
    return f'<a href="{href}" target="_self" class="{cls}">{caption}</a>'


# Header/navigation (menggantikan base.html)
nav_html = f"""
<div class="nav">
  <div class="nav-inner">
    <div class="brand"><a href="?page=home" target="_self">MangroveLeaf</a></div>
    <nav>
      {nav_link("Home", "home", page=="home")}
      {nav_link("Deteksi", "deteksi", page=="deteksi")}
      {nav_link("About", "about", page=="about")}
      {nav_link("FAQ", "faq", page=="faq")}
    </nav>
  </div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# ===== Footer helper (digunakan semua halaman) =====
def render_footer():
    st.markdown(
        '''
        <footer class="site-footer">
          <div class="inner">© 2025 Leaf Detection System. All rights reserved.</div>
        </footer>
        ''',
        unsafe_allow_html=True
    )


# ------------------------[ HOME (template/index.html) ]------------------------
if page == "home":
    # --- gambar hero: pakai file lokal "hutan mangrove.jpg" ---
    hero_local = STATIC_DIR / "hutan mangrove.jpg"
    if not hero_local.exists():
        st.warning("⚠️ File 'hutan mangrove.jpg' tidak ditemukan di folder static/. Pastikan sudah ditaruh di sana.")
    hero_src = to_data_uri(hero_local)

    # --- HERO ---
    st.markdown(f"""
    <div class="hero">
      <div class="hero-grid">
        <div>
          <h1>Sistem Klasifikasi<br/>Daun Mangrove</h1>
          <p class="lead">Sistem canggih untuk klasifikasi daun mangrove menggunakan teknologi machine learning.</p>
          <div class="btns">
            <a class="btn-primary" href="?page=deteksi" target="_self">Mulai Deteksi</a>
            <a class="btn-secondary" href="#learn-more" target="_self">Pelajari Lebih Lanjut</a>
          </div>
        </div>
      <div>
        <img src="{hero_src}" alt="Hutan mangrove" />
      </div>
    </div>
  </div>
  """, unsafe_allow_html=True)


    # --- NEWS SECTION ---
    st.markdown('<div class="section" id="learn-more">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; margin-bottom:16px;">
      <h3 style="font-size:30px; font-weight:900; color:#2C4001; margin:0;">
        Aksi Terkini Pelestarian Mangrove
      </h3>
      <p style="color:#3C4030; margin-top:6px; font-size:16px;">
        Kegiatan terbaru rehabilitasi mangrove di Indonesia
      </p>
    </div>
    """, unsafe_allow_html=True)

    news = [
    {
        "tag":"Hari Bumi 2025",
        "title":"Aksi Kolektif Selamatkan Mangrove",
        "img":"https://www.kemenlh.go.id/dashboard/public/storage/news-activities/ia4dQ6FBMfozH3DP35dqXBgrSAlV78cVmDXSajw0.jpg",
        "snippet":"KLHBPLH gaungkan gerakan nasional penyelamatan ekosistem mangrove",
        "meta":"KemenLHK",
        "link":"https://www.kemenlh.go.id/news/detail/hari-bumi-2025-klhbplh-gaungkan-aksi-kolektif-selamatkan-mangrove-indonesia"
    },
    {
        "tag":"Aksi Lokal",
        "title":"Penanaman Mangrove di Buleleng",
        "img":"https://dlh.bulelengkab.go.id/uploads/konten/40_penanaman-mangrove.jpg",
        "snippet":"DLH Kabupaten Buleleng lakukan rehabilitasi mangrove pesisir",
        "meta":"DLH Buleleng",
        "link":"https://dlh.bulelengkab.go.id/informasi/detail/berita/40_penanaman-mangrove"
    },
    {
        "tag":"Gerakan Mahasiswa",
        "title":"Penanaman Mangrove oleh Mahasiswa",
        "img":"https://lldikti5.kemdikbud.go.id/assets/images/posts/medium/tn_lldikti5_20250620183120.jpg",
        "snippet":"PMK dan HMTK ITY lakukan aksi penanaman dalam rangka Hari Lingkungan Hidup",
        "meta":"LLDIKTI V",
        "link":"https://lldikti5.kemdikbud.go.id/home/detailpost/pmk-dan-hmtk-ity-lakukan-aksi-penanaman-mangrove-dalam-rangka-hari-lingkungan-hidup"
    },
    {
        "tag":"Aksi Serentak",
        "title":"Ribuan Bibit Mangrove di Pekalongan",
        "img":"https://pekalongankota.go.id/upload/berita/berita_20250605023633.jpeg",
        "snippet":"Peringatan Hari Lingkungan Hidup Sedunia 2025 dengan penanaman massal",
        "meta":"Pemkot Pekalongan",
        "link":"https://pekalongankota.go.id/berita/aksi-serentak-peringatan-hari-lingkungan-hidup-sedunia-2025-kota-pekalongan-tanam-ribuan-bibit-mangrove-.html"
    },
    {
        "tag":"Aksi Bersih-bersih",
        "title":"Bersih-bersih Kawasan Mangrove",
        "img":"https://prokopim.bengkaliskab.go.id/gambar/images/IMG-20250621-WA0011.jpg",
        "snippet":"PT KPI RU II Sungai Pakning gelar aksi bersih-bersih sempena Hari Lingkungan Hidup",
        "meta":"Bengkalis",
        "link":"https://prokopim.bengkaliskab.go.id/web/detailberita/16788/sempena-hari-lingkungan-hidup-sedunia-tahun-2025,-pt-kpi-ru-ii-sungai-pakning-gelar-bersih-bersih"
    },
]

    # grid kartu
    # ===== RENDER NEWS (1 baris scrollable, 5 kartu terlihat) =====
    items_html = []
    for n in news:
        items_html.append(
            f'''<div class="news-item">
      <div class="card-news">
        <a href="{n['link']}" target="_blank" style="text-decoration:none;">
          <img src="{n['img']}" alt="{n['title']}">
          <div class="card-body">
            <span class="badge">{n['tag']}</span>
            <div class="card-title">{n['title']}</div>
            <div class="card-snippet">{n['snippet']}</div>
            <div class="card-footer">
              <small>{n['meta']}</small>
              <span class="cta">Baca →</span>
            </div>
          </div>
        </a>
      </div>
    </div>'''
        )

    scroller_html = '<div class="news-row"><div class="news-scroller">' + ''.join(items_html) + '</div></div>'
    st.markdown(scroller_html, unsafe_allow_html=True)
    st.markdown('<div class="news-rail"></div>', unsafe_allow_html=True)

    # --- FOOTER ---
    render_footer()

# ------------------------[ DETEKSI (template/deteksi.html) ]-------------------
elif page == "deteksi":
    # Form unggah & opsi (mirip form di deteksi.html)
    st.markdown('<section class="card"><h2>Deteksi Daun</h2>', unsafe_allow_html=True)

    left, right = st.columns([2,1])
    with left:
        uploaded_files = st.file_uploader(
            "Pilih gambar (bisa multiple):",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True
        )
    # Toggle ini menulis ke variabel global karena berada di scope modul (bukan di dalam function)
    with right:
        show_features = st.checkbox("Tampilkan ringkasan fitur", value=False, help="Menampilkan subset fitur kunci seperti di template.")
        USE_MASKED_INPUT = st.checkbox("Masked background ke putih", value=True, help="Membantu robust terhadap variasi latar.")

    go = st.button("Prediksi", type="primary", disabled=(not uploaded_files))
    st.markdown('</section>', unsafe_allow_html=True)

    error = None
    results = []
    rows_for_csv = []
    csv_ready = False

    if go:
        if not uploaded_files:
            error = "Pilih minimal satu gambar."
        else:
            with st.spinner("Memproses..."):
                for f in uploaded_files:
                    try:
                        data = np.frombuffer(f.read(), np.uint8)
                        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        if bgr is None:
                            raise ValueError("Gagal membaca gambar (format tidak didukung).")

                        # Prediksi
                        pred, conf, topk = predict_bgr(bgr)

                        # Fitur + anotasi
                        feats_all = extract_manual_features_from_bgr(bgr)
                        anno = annotated_with_features(bgr, pred, conf, feats_all, draw_label_on_image=True)
                        rgb_anno = cv2.cvtColor(anno, cv2.COLOR_BGR2RGB)

                        # Tampilkan satu item hasil (gambar + metadata)
                        st.markdown('<section class="card">', unsafe_allow_html=True)
                        cols = st.columns([1,1])
                        with cols[0]:
                            st.image(rgb_anno, caption=f"{f.name}")
                        with cols[1]:
                            st.markdown(f"**Prediksi:** <b>{pred}</b> ({conf:.3f})", unsafe_allow_html=True)
                            st.markdown("**Top-k:** " + ", ".join([f"{lbl}:{p:.3f}" for lbl, p in topk]))
                            slug = label_to_slug(pred)
                            if slug: 
                              st.markdown(
                                f'<div style="margin-top:8px">'
                                f'  <a class="btn-primary" href="?page=species&sp={slug}" target="_self">'
                                f'    🔎 Lihat detail spesies'
                                f'  </a>'
                                f'</div>',
                                unsafe_allow_html=True
                              )
                            if show_features:
                                # Tabel mini (subset KEY_FEATURES)
                                lines = []
                                for k in KEY_FEATURES:
                                    if k in feats_all:
                                        lines.append((k, f"{feats_all[k]:.4f}"))
                                if lines:
                                    mini = "<table class='mini'>" + "".join(
                                        [f"<tr><td>{k}</td><td style='text-align:right'>{v}</td></tr>" for k,v in lines]
                                    ) + "</table>"
                                    st.markdown(mini, unsafe_allow_html=True)
                        st.markdown('</section>', unsafe_allow_html=True)

                        # Data untuk CSV (pakai FEATURE_COLS agar konsisten dgn model hybrid)
                        row = {"filename": f.name, "pred": pred, "confidence": float(conf)}
                        for c in FEATURE_COLS:
                            row[c] = float(feats_all.get(c, 0.0))
                        rows_for_csv.append(row)

                        # Ringkasan tabel kecil
                        results.append({
                            "filename": f.name,
                            "pred": pred,
                            "conf": f"{conf:.3f}",
                            "topk": [f"{lbl}:{p:.3f}" for lbl, p in topk],
                        })

                    except Exception as e:
                        st.error(f"{f.name}: {e}")

            csv_ready = len(rows_for_csv) > 0

    # Error
    if error:
        st.markdown(f"<div class='error'>{error}</div>", unsafe_allow_html=True)

    # Tombol unduh CSV (gantikan link download di template Flask)
    if csv_ready:
        df_csv = pd.DataFrame(rows_for_csv)
        csv_bytes = df_csv.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV fitur",
            data=csv_bytes,
            file_name=f"features_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
            mime="text/csv",
        )
    render_footer()

elif page == "about":
    import mimetypes, base64

    # --- cari file di static/images ATAU static, coba beberapa ekstensi umum ---
    EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    def find_img(stem_or_name: str) -> Path:
        # kalau nama sudah pakai ekstensi, pakai langsung; kalau belum, coba semua EXTS
        cand_names = [stem_or_name] if Path(stem_or_name).suffix else [stem_or_name + e for e in EXTS]
        folders = [STATIC_DIR / "images", STATIC_DIR]  # urutan pencarian
        for nm in cand_names:
            for folder in folders:
                p = folder / nm
                if p.exists():
                    return p
        # fallback: scan seluruh static (misal nama sedikit beda huruf besar/kecil)
        hits = list(STATIC_DIR.rglob(stem_or_name + "*"))
        return hits[0] if hits else (STATIC_DIR / (stem_or_name + ".jpg"))

    # --- ubah file menjadi data URI agar <img> bisa menampilkannya ---
    def to_data_uri(p: Path, ph_text: str = "") -> str:
        if not p.exists():
            # placeholder SVG bila file tak ada
            txt = ph_text or f"Missing: {p.name}"
            svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500"><rect width="100%" height="100%" fill="#eef3ee"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#365902" font-size="20" font-family="sans-serif">{txt}</text></svg>'
            return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        b64  = base64.b64encode(p.read_bytes()).decode()
        return f"data:{mime};base64,{b64}"

    # -- daftar gambar (cukup tulis stem/namanya saja; taruh di static/ atau static/images/)
    ekos = find_img("daun pohon mangrove")
    daun = find_img("daun mangrove")
    am   = find_img("daun am")
    bg   = find_img("daun bg")
    hl   = find_img("daun hl")
    ao   = find_img("daun ao")
    lt   = find_img("daun lt")
    ra   = find_img("daun ra")
    sa   = find_img("daun sa")

    # info bila ada yang hilang
    missing = [p for p in [ekos,daun,am,bg,hl,ao,lt,ra,sa] if not p.exists()]
    if missing:
        st.warning("File tidak ditemukan di folder static:\n- " + "\n- ".join(str(m) for m in missing))

    st.markdown('<div class="about-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="about-title">Tentang Mangrove</h1>', unsafe_allow_html=True)
    # Ekosistem Mangrove
    st.markdown(f"""
    <div class="ekos-grid">
      <div class="ekos-img">
        <img src="{to_data_uri(ekos)}" alt="Ekosistem Mangrove"/>
        <div class="ekos-badge">Ekosistem Pesisir Indonesia</div>
      </div>
      <div class="ekos-card">
        <h3>Ekosistem Mangrove</h3>
        <p style="color:#3C4030;margin:0 0 6px;">
          Mangrove merupakan komunitas tumbuhan pantai yang memiliki ketahanan tinggi terhadap kadar garam
          dan membentuk ekosistem khas di wilayah pesisir tropis dan subtropis.
        </p>
        <ul>
          <li>Berfungsi sebagai habitat penting bagi biota laut dalam siklus hidupnya</li>
          <li>Memiliki sistem perakaran kompleks yang menstabilkan sedimentasi pantai</li>
          <li>Berperan sebagai filter alami polutan dan penyerap karbon dioksida</li>
        </ul>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Ciri Khas Daun
    st.markdown(f"""
    <h2 class="about-heading" style="font-size:34px;margin:18px 0 12px;color:#2C4001;">Ciri Khas Daun Mangrove</h2>
    <div class="leaf-grid">
      <div class="leaf-card">
        <div class="sec">
          <h5>Bentuk Daun</h5>
          <ul>
            <li>Umumnya lonjong/elips dengan ujung runcing (adaptasi mengalirkan garam)</li>
            <li>Beberapa spesies bulat tebal (<em>Sonneratia</em>) atau kecil (<em>Lumnitzera</em>)</li>
          </ul>
        </div>
      <div class="sec">
        <h5>Warna Daun</h5>
        <ul>
          <li>Dominan hijau tua (permukaan atas)</li>
          <li>Bagian bawah lebih pucat/kekuningan (akumulasi garam)</li>
        </ul>
      </div>
      <div class="sec">
        <h5>Tekstur Daun</h5>
        <ul>
          <li>Tebal dan berdaging (sukulen) untuk menyimpan air</li>
          <li>Permukaan licin/berkulit untuk mengurangi penguapan</li>
          <li>Beberapa spesies berbulu halus atau kasar</li>
        </ul>
      </div>
    </div>
    <div>
      <img class="leaf-img" src="{to_data_uri(daun)}" alt="Daun Mangrove"/>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Jenis-Jenis (pakai file lokal juga)
    species = [
        ("Avicennia Marina", am, ["Bentuk: Bulat telur, ujung tumpul","Warna: Hijau keabu-abuan","Tekstur: Sukulen dengan bintik garam"]),
        ("Bruguiera Gymnorhiza", bg, ["Bentuk: Elips lancip","Warna: Hijau tua mengilap","Tekstur: Kaku seperti kulit"]),
        ("Heritiera Littoralis", hl, ["Bentuk: Lonjong dengan ujung runcing","Warna: Hijau metalik","Tekstur: Kertas tipis"]),
        ("Avicennia Officinalis", ao, ["Bentuk: Lonjong lebar","Warna: Hijau kekuningan","Tekstur: Sukulen, permukaan kasar"]),
        ("Lumnitzera Littorea", lt, ["Bentuk: Bulat kecil","Warna: Hijau tua","Tekstur: Keras dan tebal"]),
        ("Rhizophora Apiculata", ra, ["Bentuk: Elips meruncing","Warna: Hijau tua","Tekstur: Licin dan kaku"]),
        ("Sonneratia Alba", sa, ["Bentuk: Bulat tebal","Warna: Hijau muda","Tekstur: Berdaging sukulen"]),
    ]

    cards = []
    for title, imguri, points in species:
        bullets = "".join(f"<li>{p}</li>" for p in points)
        card_html = dedent(f"""
        <div class="card-spec">
        <img src="{to_data_uri(imguri)}" alt="Daun {title}" loading="lazy"/>
        <div class="body">
        <h5>{title}</h5>
        <ul style="margin:0 0 0 18px;color:#3C4030;">{bullets}</ul>
        </div>
        </div>
        """).strip()
        
        # >>> NEW: link khusus untuk Avicennia Marina
        slug = title.lower().replace(" ", "_")
        if slug in ("avicennia_marina", "bruguiera_gymnorhiza", "heritiera_littoralis", "avicennia_officinalis", "lumnitzera_littorea", "rhizophora_apiculata", "sonneratia_alba"):
            card_html = f'<a href="?page=species&sp={slug}" target="_self" style="text-decoration:none;color:inherit;">{card_html}</a>'
        cards.append(card_html)

    # (TAMBAHKAN) — pengganti dua blok di atas
    cards_html = ''.join(cards)

    carousel_component = f"""
    <div id='carousel-root'>
      <style>
        .carousel {{ position:relative; max-width:1240px; margin:0 auto 6px; font-family:inherit; }}
        .title    {{text-align:center; color:#2C4001; font-weight:800; font-size:34px; margin:18px 0 12px; }}
        .subtle   {{ text-align:center; color:#3C4030; font-weight:400; margin:6px 0 18px; font-size:16px; }}

        .track {{
          display:flex; flex-wrap:nowrap; gap:32px; padding:0 64px 18px;
          overflow-x:auto; scroll-snap-type:x proximity; -webkit-overflow-scrolling:touch;
          scrollbar-width:thin;
        }}
        .track::-webkit-scrollbar{{ height:8px; }}
        .track::-webkit-scrollbar-thumb{{ background:#365902; border-radius:10px; }}

        .card-spec{{
          flex:0 0 320px; min-width:320px; background:#fff; border:1px solid #cfd6c6;
          border-radius:12px; overflow:hidden; box-shadow:0 6px 18px rgba(0,0,0,.08);
          scroll-snap-align:start;
        }}
        .card-spec img{{ width:100%; aspect-ratio:4/3; object-fit:cover; display:block; }}
        .card-spec h5{{ color:#2C4001; margin:2px 0 10px; font-size:20px; font-weight:800; padding:center; }}
        .card-spec .body{{ padding:0 16px 16px; }}
        .card-spec ul{{ margin:0 0 0 18px; color:#3C4030; font-size:16px; line-height:1.5; }}
        .card-spec li{{  margin:.25rem 0;    /* jarak antar poin */
        }}
        .cbtn{{
          position:absolute; top:50%; transform:translateY(-50%);
          width:44px; height:44px; border-radius:50%;
          display:flex; align-items:center; justify-content:center;
          background:#365902; color:#fff; border:2px solid #fff;
          box-shadow:0 2px 10px rgba(0,0,0,.15); cursor:pointer;
        }}
        .cbtn.left{{ left:-22px; }}      /* sama seperti styling global kamu */
        .cbtn.right{{ right:-22px; }}
        @media (max-width:600px){{ .cbtn{{ display:none; }} }}
      </style>

      <div class="title">Jenis-Jenis Mangrove</div>
      <div class="subtle">Geser untuk melihat lebih banyak jenis</div>

      <div class="carousel">
        <div id="btnLeft" class="cbtn left">‹</div>
        <div id="btnRight" class="cbtn right">›</div>
        <div id="track" class="track">{cards_html}</div>
      </div>

      <script>
        const t = document.getElementById('track');
        const l = document.getElementById('btnLeft');
        const r = document.getElementById('btnRight');

        function step() {{
          const first = t.querySelector('.card-spec');
          const gap = parseInt(getComputedStyle(t).gap || '24', 10);
          return (first ? first.getBoundingClientRect().width : 320) + gap;
        }}
        function by(dir) {{ t && t.scrollBy({{ left: dir * step(), behavior: 'smooth' }}); }}

        l && l.addEventListener('click', () => by(-1));
        r && r.addEventListener('click', () => by( 1));

        // Scroll horizontal pakai roda mouse/trackpad
        t && t.addEventListener('wheel', (e) => {{
          if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {{
            e.preventDefault();
            t.scrollLeft += e.deltaY;
          }}
        }}, {{ passive: false }});
      </script>
    </div>
    """

    html(carousel_component, height=560, scrolling=False)


    render_footer()

elif page == "species":
    # ------- Pilih spesies via query param -------
    sp = st.query_params.get("sp", "avicennia_marina")
    if isinstance(sp, list): 
        sp = sp[0] if sp else "avicennia_marina"

    # ------- Data dua spesies -------
    def get_species(sp_key: str):
        if sp_key == "avicennia_marina":
            return {
                "nama_latin": "Avicennia marina",
                "nama_lokal": [
                    "Api-api putih","Api-api abang","Sia-sia putih","Sie-sie",
                    "Pejapi","Nyapi","Hajusia","Pai"
                ],
                "deskripsi_umum": (
                    "Pohon/belukar selalu hijau yang dapat mencapai tinggi ±30 m. "
                    "Memiliki sistem perakaran horizontal yang rumit dengan banyak pneumatofor "
                    "(akar nafas) tegak seperti pensil. Kulit kayu halus, abu-abu kehijauan dan "
                    "mudah terkelupas menjadi kepingan kecil. Ranting muda/tangkai daun kekuningan, tidak berbulu."
                ),
                "karakteristik": [
                    "Akar nafas (pneumatofor) rapat dan tegak",
                    "Daun memiliki kelenjar garam (bintik cekung halus)",
                    "Bunga kecil kekuningan; buah dapat dimakan",
                ],
                "daun": {
                    "bentuk": "Elips sampai bulat telur terbalik",
                    "ukuran": "± 9 × 4,5 cm (variatif)",
                    "warna": "Hijau keabu-abuan (atas mengilap, bawah lebih pucat)",
                    "gagang": "± 0,5–1 cm",
                    "letak": "Sederhana, berhadapan",
                    "deskripsi": (
                        "Permukaan atas sering tampak berbintik (kelenjar garam). "
                        "Ujung daun meruncing sampai membundar; tebal-sukulen."
                    ),
                },
                "manfaat": [
                    "Kayu cocok untuk arang/kayu bakar",
                    "Ekstrak daun digunakan dalam pengobatan tradisional (kulit terbakar)",
                    "Resin kulit kayu digunakan secara tradisional sebagai kontrasepsi",
                    "Buah dapat dimakan (setempat, setelah pengolahan)",
                    "Kayu untuk bahan kertas berkualitas",
                    "Daun sebagai pakan ternak di beberapa daerah",
                ],
                "penyebaran": {
                    "lokasi": [
                        "Aceh","Sumatera Utara","Sumatera Barat","Riau","Kepulauan Riau",
                        "Jawa","Bali","Nusa Tenggara","Kalimantan Barat","Kalimantan Timur",
                        "Sulawesi","Maluku","Papua"
                    ],
                    "koordinat": [
                        {"nama":"Aceh","lat":4.695135,"lng":96.749399},
                        {"nama":"Sumatera Utara","lat":2.115354,"lng":99.545097},
                        {"nama":"Sumatera Barat","lat":-0.739939,"lng":100.800005},
                        {"nama":"Riau","lat":0.293347,"lng":101.706829},
                        {"nama":"Kepulauan Riau","lat":0.906,"lng":104.142},
                        {"nama":"Jawa","lat":-6.174465,"lng":106.822745},
                        {"nama":"Bali","lat":-8.409518,"lng":115.188919},
                        {"nama":"Nusa Tenggara","lat":-8.652497,"lng":121.079371},
                        {"nama":"Kalimantan Barat","lat":-0.0227,"lng":109.344},
                        {"nama":"Kalimantan Timur","lat":-0.502106,"lng":117.153709},
                        {"nama":"Sulawesi","lat":-3.549121,"lng":121.727539},
                        {"nama":"Maluku","lat":-3.238462,"lng":130.145273},
                        {"nama":"Papua","lat":-2.533333,"lng":140.716667},
                    ],
                },
            }
        elif sp_key == "bruguiera_gymnorhiza":
            return {
                "nama_latin": "Bruguiera gymnorhiza",
                "nama_lokal": [
                    "Pertut","Taheup","Tenggel","Putut","Tumu","Tomo",
                    "Kandeka","Tanjang merah","Tanjang","Lindur","Sala-sala",
                    "Dau","Tongke","Totongkek","Mutut besar","Wako","Bako",
                    "Bangko","Mangi-mangi","Sarau"
                ],
                "deskripsi_umum": (
                    "Pohon selalu hijau kadang mencapai 30 m. Kulit kayu berlentisel, permukaan halus "
                    "hingga kasar berwarna abu-abu tua sampai coklat. Akar papan melebar di pangkal, "
                    "juga memiliki sejumlah akar lutut."
                ),
                "karakteristik": [
                    "Akar papan melebar pada pangkal batang",
                    "Memiliki akar lutut (knee roots)",
                    "Kulit kayu dengan lentisel; warna abu-abu tua–coklat",
                ],
                "daun": {
                    "deskripsi": (
                        "Daun berkulit; hijau tua di permukaan atas dan hijau kekuningan di bawah, "
                        "sering dengan bercak hitam halus."
                    ),
                    "letak": "Sederhana & berlawanan",
                    "bentuk": "Elips sampai elips-lanset",
                    "ujung": "Meruncing",
                    "ukuran": "4,5–7 × 8,5–22 cm",
                    "warna": "Hijau tua (atas), hijau kekuningan (bawah)",
                    "gagang": "—"
                },
                "manfaat": [
                    "Bagian dalam hipokotil dapat dimakan (manisan kandeka) setelah diolah",
                    "Kayu merah digunakan sebagai kayu bakar",
                    "Bahan arang berkualitas tinggi",
                    "Ekstrak kulit digunakan dalam pengobatan tradisional",
                    "Daun muda digunakan sebagai pakan ternak"
                ],
                "penyebaran": {
                    "lokasi": [
                        "Aceh","Sumatera Utara","Riau","Kepulauan Riau",
                        "Jawa Barat","Jawa Timur","Bali",
                        "Kalimantan Barat","Kalimantan Timur",
                        "Sulawesi Selatan","Sulawesi Tenggara",
                        "Maluku","Papua Barat","Papua"
                    ],
                    "koordinat": [
                        {"nama":"Aceh","lat":4.695135,"lng":96.749399},
                        {"nama":"Sumatera Utara","lat":2.115354,"lng":99.545097},
                        {"nama":"Riau","lat":0.293347,"lng":101.706829},
                        {"nama":"Kepulauan Riau","lat":3.945651,"lng":108.142867},
                        {"nama":"Jawa Barat","lat":-6.914744,"lng":107.609810},
                        {"nama":"Jawa Timur","lat":-7.245972,"lng":112.737991},
                        {"nama":"Bali","lat":-8.409518,"lng":115.188919},
                        {"nama":"Kalimantan Barat","lat":-0.278781,"lng":111.475285},
                        {"nama":"Kalimantan Timur","lat":-0.502106,"lng":117.153709},
                        {"nama":"Sulawesi Selatan","lat":-5.147665,"lng":119.432731},
                        {"nama":"Sulawesi Tenggara","lat":-3.549121,"lng":121.727539},
                        {"nama":"Maluku","lat":-3.238462,"lng":130.145273},
                        {"nama":"Papua Barat","lat":-1.336115,"lng":133.174716},
                        {"nama":"Papua","lat":-2.533333,"lng":140.716667}
                    ]
                }
            }
        elif sp_key == "heritiera_littoralis":
            return {
                "nama_latin": "Heritiera littoralis",
                "nama_lokal": [
                    "Dungu","Dungun","Atung laut","Lawanan kete","Rumung",
                    "Balang pasisir","Lawang","Cerlang laut","Lulun","Rurun",
                    "Belohila","Blakangabu","Bayur laut"
                ],
                "deskripsi_umum": (
                    "Pohon selalu hijau hingga ±25 m. Akar papan berkembang sangat jelas. "
                    "Kulit kayu gelap/abu-abu, bersisik dan bercelah. Individu pohon "
                    "biasanya berumah dua (bunga jantan atau betina pada individu berbeda)."
                ),
                "karakteristik": [
                    "Akar papan (buttress roots) jelas dan besar",
                    "Kulit kayu bersisik serta bercelah",
                    "Bunga uniseksual (terpisah jantan/betina)"
                ],
                "daun": {
                    "deskripsi": (
                        "Daun kukuh, berkulit, sering berkelompok di ujung cabang. "
                        "Permukaan atas hijau gelap, bagian bawah putih-keabu akibat lapisan halus."
                    ),
                    "gagang": "0,5–2 cm",
                    "letak": "Sederhana, bersilangan",
                    "bentuk": "Bulat telur–elips",
                    "ujung": "Meruncing",
                    "ukuran": "10–20 × 5–10 cm (kadang 30 × 15–18 cm)",
                    "warna": "Hijau gelap (atas), putih-keabu-abuan (bawah)"
                },
                "manfaat": [
                    "Kayu bakar yang baik",
                    "Kayu tahan lama untuk perahu, rumah, dan tiang telepon",
                    "Buah untuk mengobati diare dan disentri",
                    "Biji digunakan dalam pengolahan ikan",
                    "Ekstrak daun dipakai dalam pengobatan tradisional"
                ],
                "penyebaran": {
                    "lokasi": [
                        "Aceh","Sumatera Utara","Sumatera Barat","Riau","Kepulauan Riau",
                        "Jawa","Bali","Nusa Tenggara","Kalimantan Barat","Kalimantan Timur",
                        "Sulawesi","Maluku","Papua"
                ],
                "koordinat": [
                    {"nama":"Aceh","lat":4.695135,"lng":96.749399},
                    {"nama":"Sumatera Utara","lat":2.115354,"lng":99.545097},
                    {"nama":"Sumatera Barat","lat":-0.739939,"lng":100.800005},
                    {"nama":"Riau","lat":0.293347,"lng":101.706829},
                    {"nama":"Kepulauan Riau","lat":3.945651,"lng":108.142867},
                    {"nama":"Jawa","lat":-7.245972,"lng":112.737991},
                    {"nama":"Bali","lat":-8.409518,"lng":115.188919},
                    {"nama":"Nusa Tenggara","lat":-8.652933,"lng":117.361648},
                    {"nama":"Kalimantan Barat","lat":-0.278781,"lng":111.475285},
                    {"nama":"Kalimantan Timur","lat":-0.502106,"lng":117.153709},
                    {"nama":"Sulawesi","lat":-3.549121,"lng":121.727539},
                    {"nama":"Maluku","lat":-3.238462,"lng":130.145273},
                    {"nama":"Papua","lat":-2.533333,"lng":140.716667}
                ]
              }
            }
        elif sp_key == "avicennia_officinalis":
            return {
                "nama_latin": "Avicennia officinalis",
                "nama_lokal": [
                    "Api-api","Api-api daun lebar","Api-api ludat",
                    "Sia-sia putih","Papi","Api-api kacang",
                    "Merahu","Marahuf"
                ],
                "deskripsi_umum": (
                    "Pohon, biasanya hingga ±12 m (kadang sampai ±20 m). Umumnya memiliki "
                    "akar tunjang dan akar nafas tipis berbentuk jari dengan banyak lentisel. "
                    "Kulit kayu luar halus berwarna hijau-keabu-abuan hingga abu-abu-kecoklatan, "
                    "sering tampak lentisel."
                ),
                "karakteristik": [
                    "Akar tunjang + pneumatofor (akar nafas) dengan lentisel",
                    "Kulit kayu halus, hijau-keabu-abuan",
                    "Tinggi dapat mencapai 12–20 m"
                ],
                "daun": {
                    "deskripsi": (
                        "Atas hijau tua, bawah hijau-kekuningan/abu-abu kehijauan. "
                        "Permukaan atas bertabur bintik kelenjar berbentuk cekung."
                    ),
                    "unit_letak": "Sederhana & berlawanan",
                    "bentuk": "Bulat telur terbalik; bulat memanjang–bulat telur terbalik; atau elips bulat memanjang",
                    "ujung": "Membundar, menyempit ke arah gagang",
                    "ukuran": "± 12,5 × 6 cm",
                    "warna": "Hijau tua (atas), hijau-kekuningan/abu-abu kehijauan (bawah)",
                    "gagang": "—"
                },
                "manfaat": [
                    "Buah dapat dimakan",
                    "Kayu untuk kayu bakar",
                    "Getah kayu digunakan sebagai bahan alat kontrasepsi (tradisional)",
                    "Daun dipakai dalam pengobatan tradisional",
                    "Akar membantu stabilisasi sedimen pantai"
                ],
                "penyebaran": {
                    "lokasi": [
                        "Aceh","Sumatera Utara","Riau","Jambi","Sumatera Selatan","Lampung",
                        "Jawa","Bali","Kalimantan Barat","Kalimantan Timur",
                        "Sulawesi Selatan","Sulawesi Tenggara","Maluku","Papua"
                    ],
                    "koordinat": [
                        {"nama":"Aceh","lat":4.695135,"lng":96.749399},
                        {"nama":"Sumatera Utara","lat":2.115354,"lng":99.545097},
                        {"nama":"Riau","lat":0.293347,"lng":101.706829},
                        {"nama":"Jambi","lat":-1.485183,"lng":102.438058},
                        {"nama":"Sumatera Selatan","lat":-2.990934,"lng":104.756556},
                        {"nama":"Lampung","lat":-5.109730,"lng":105.547266},
                        {"nama":"Jawa","lat":-7.245972,"lng":112.737991},
                        {"nama":"Bali","lat":-8.409518,"lng":115.188919},
                        {"nama":"Kalimantan Barat","lat":-0.278781,"lng":111.475285},
                        {"nama":"Kalimantan Timur","lat":-0.502106,"lng":117.153709},
                        {"nama":"Sulawesi Selatan","lat":-5.147665,"lng":119.432731},
                        {"nama":"Sulawesi Tenggara","lat":-3.549121,"lng":121.727539},
                        {"nama":"Maluku","lat":-3.238462,"lng":130.145273},
                        {"nama":"Papua","lat":-2.533333,"lng":140.716667}
                    ]
                }
            }
        elif sp_key == "lumnitzera_littorea":
           return {
               "nama_latin": "Lumnitzera littorea",
               "nama_lokal": [
                   "Teruntum merah","Api-api uding","Sesop","Sesak","Geriting",
                   "Randai","Riang laut","Taruntung","Duduk agung","Duduk gedeh",
                   "Welompelong","Posi-posi","Ma gorago","Kedukduk"
                ],
                "deskripsi_umum": (
                    "Pohon selalu hijau, tumbuh tersebar, tinggi dapat mencapai ±25 m (umumnya lebih "
                    "rendah). Memiliki akar nafas berbentuk lutut berwarna coklat tua. Kulit kayu "
                    "bercelah/retak memanjang (longitudinal)."
                ),
                "karakteristik": [
                    "Akar nafas (knee roots) berwarna coklat tua",
                    "Kulit kayu bercelah membujur",
                    "Tinggi pohon dapat mencapai ±25 m"
                ],
                "daun": {
                    "deskripsi": "Daun agak tebal berdaging, keras/kaku, berumpun pada ujung dahan.",
                    "gagang": "≤ 5 mm",  # dari 'tangkai' → disamakan ke kunci 'gagang'
                    "letak": "Sederhana, bersilangan; berumpun di ujung dahan",
                    "bentuk": "Bulat telur terbalik",
                    "ujung": "Membundar",
                    "ukuran": "2–8 × 1–2,5 cm",
                    "warna": "Hijau tua mengilap (tebal & kaku)"
                },
                "manfaat": [
                    "Kayu kuat dan sangat tahan terhadap air",
                    "Cocok untuk lemari & furnitur",
                    "Beraroma wangi menyerupai mawar",
                    "Bahan kerajinan kayu bernilai tinggi",
                    "Ekstrak kulit dipakai dalam pengobatan tradisional"
                ],
                "penyebaran": {
                    "lokasi": [
                        "Aceh","Sumatera Utara","Riau","Kepulauan Riau",
                        "Jawa Barat","Jawa Timur","Bali",
                        "Kalimantan Barat","Kalimantan Timur",
                        "Sulawesi Utara","Sulawesi Selatan",
                        "Maluku","Papua Barat","Papua"
                    ],
                    "koordinat": [
                        {"nama":"Aceh","lat":4.695135,"lng":96.749399},
                        {"nama":"Sumatera Utara","lat":2.115354,"lng":99.545097},
                        {"nama":"Riau","lat":0.293347,"lng":101.706829},
                        {"nama":"Kepulauan Riau","lat":3.945651,"lng":108.142867},
                        {"nama":"Jawa Barat","lat":-6.914744,"lng":107.609810},
                        {"nama":"Jawa Timur","lat":-7.245972,"lng":112.737991},
                        {"nama":"Bali","lat":-8.409518,"lng":115.188919},
                        {"nama":"Kalimantan Barat","lat":-0.278781,"lng":111.475285},
                        {"nama":"Kalimantan Timur","lat":-0.502106,"lng":117.153709},
                        {"nama":"Sulawesi Utara","lat":1.474830,"lng":124.842079},
                        {"nama":"Sulawesi Selatan","lat":-5.147665,"lng":119.432731},
                        {"nama":"Maluku","lat":-3.238462,"lng":130.145273},
                        {"nama":"Papua Barat","lat":-1.336115,"lng":133.174716},
                        {"nama":"Papua","lat":-2.533333,"lng":140.716667}
                    ]
                }
            }
        elif sp_key == "rhizophora_apiculata":
            return {
                "nama_latin": "Rhizophora apiculata",
                "nama_lokal": [
                    "Bakau minyak","Bakau tandok","Bakau akik","Bakau puteh",
                    "Bakau kacang","Bakau leutik","Akik","Bangka minyak",
                    "Donggo akit","Jankar","Abat","Parai","Mangi-mangi",
                    "Slengkreng","Tinjang","Wako"
                ],
                "deskripsi_umum": (
                    "Pohon hingga ±30 m (Ø batang ~50 cm). Memiliki sistem perakaran khas "
                    "dapat mencapai ±5 m; kadang muncul akar udara dari cabang. Kulit kayu "
                    "abu-abu tua dan bervariasi."
                ),
                "karakteristik": [
                    "Akar penyangga/udara yang menonjol (hingga ±5 m)",
                    "Kadang ada akar udara keluar dari cabang",
                    "Tinggi pohon hingga ±30 m, diameter ~50 cm",
                ],
                "daun": {
                    "deskripsi": "Daun berkulit; hijau tua, bagian tengah sering hijau muda, sisi bawah cenderung kemerahan.",
                    "gagang": "17–35 mm, kemerahan",
                    "letak": "Sederhana & berlawanan",
                    "bentuk": "Elips menyempit",
                    "ujung": "Meruncing",
                    "ukuran": "7–19 × 3,5–8 cm",
                    "warna": "Hijau tua (tengah hijau muda; bawah kemerahan)"
                },
                "manfaat": [
                    "Kayu untuk bahan bangunan, kayu bakar, dan arang",
                    "Kulit kayu kaya tanin (hingga ~30%)",
                    "Cabang akar dipakai sebagai jangkar",
                    "Penahan pematang tambak & penghijauan pesisir",
                    "Menyediakan habitat penting bagi biota pesisir"
                ],
                "penyebaran": {
                    "lokasi": [
                        "Aceh","Sumatera Utara","Sumatera Barat","Riau",
                        "Jambi","Sumatera Selatan","Lampung",
                        "Banten","Jakarta","Jawa Barat","Jawa Tengah",
                        "Jawa Timur","Bali","Kalimantan Barat",
                        "Kalimantan Timur","Sulawesi Selatan",
                        "Sulawesi Tenggara","Maluku","Papua"
                    ],
                    "koordinat": [
                        {"nama":"Aceh","lat":4.695135,"lng":96.749399},
                        {"nama":"Sumatera Utara","lat":2.115354,"lng":99.545097},
                        {"nama":"Sumatera Barat","lat":-0.739939,"lng":100.800005},
                        {"nama":"Riau","lat":0.293347,"lng":101.706829},
                        {"nama":"Jambi","lat":-1.485183,"lng":102.438058},
                        {"nama":"Sumatera Selatan","lat":-2.990934,"lng":104.756556},
                        {"nama":"Lampung","lat":-5.109730,"lng":105.547266},
                        {"nama":"Banten","lat":-6.405817,"lng":106.064018},
                        {"nama":"Jakarta","lat":-6.208763,"lng":106.845599},
                        {"nama":"Jawa Barat","lat":-6.914744,"lng":107.609810},
                        {"nama":"Jawa Tengah","lat":-6.966667,"lng":110.416664},
                        {"nama":"Jawa Timur","lat":-7.245972,"lng":112.737991},
                        {"nama":"Bali","lat":-8.409518,"lng":115.188919},
                        {"nama":"Kalimantan Barat","lat":-0.278781,"lng":111.475285},
                        {"nama":"Kalimantan Timur","lat":-0.502106,"lng":117.153709},
                        {"nama":"Sulawesi Selatan","lat":-5.147665,"lng":119.432731},
                        {"nama":"Sulawesi Tenggara","lat":-3.549121,"lng":121.727539},
                        {"nama":"Maluku","lat":-3.238462,"lng":130.145273},
                        {"nama":"Papua","lat":-2.533333,"lng":140.716667},
                    ]
                }
            }
        elif sp_key == "sonneratia_alba":
            return {
                "nama_latin": "Sonneratia alba",
                "nama_lokal": [
                    "Pedada","Perepat","Pidada","Bogem","Bidada","Posi-posi","Wahat","Putih",
                    "Beropak","Bangka","Susup","Kedada","Muntu","Sopo","Barapak","Pupat","Mange-mange"
                ],
                "deskripsi_umum": (
                    "Pohon selalu hijau, tumbuh tersebar, tinggi umumnya hingga ±15 m. "
                    "Kulit kayu putih tua hingga coklat dengan celah longitudinal halus. "
                    "Akar berbentuk kabel di bawah tanah dan muncul sebagai akar nafas "
                    "berbentuk kerucut tumpul (hingga ±25 cm)."
                ),
                "karakteristik": [
                    "Akar nafas (pneumatofor) kerucut tumpul setinggi ±25 cm",
                    "Kulit kayu putih tua–coklat dengan celah memanjang",
                    "Tinggi pohon dapat mencapai ±15 m",
                ],
                "daun": {
                    "deskripsi": "Daun berkulit; ada kelenjar tidak berkembang di pangkal gagang daun.",
                    "gagang": "6–15 mm",
                    "letak": "Sederhana & berlawanan",
                    "bentuk": "Bulat telur terbalik",
                    "ujung": "Membundar",
                    "ukuran": "5–12,5 × 3–9 cm",
                },
                "manfaat": [
                    "Buah asam dapat dimakan",
                    "Kayu untuk perahu dan bahan bangunan",
                    "Sumber bahan bakar alternatif",
                    "Akar nafas dimanfaatkan sebagai gabus/pelampung",
                    "Bunga jadi sumber nektar lebah madu",
                    "Daun muda untuk pakan ternak",
                ],
                "penyebaran": {
                    "lokasi": [
                        "Aceh","Sumatera Utara","Riau","Kepulauan Riau","Jakarta","Jawa Barat","Jawa Timur","Bali",
                        "Kalimantan Barat","Kalimantan Timur","Sulawesi Utara","Sulawesi Selatan","Sulawesi Tenggara",
                        "Maluku","Papua Barat","Papua"
                    ],
                    "koordinat": [
                        {"nama":"Aceh","lat":4.695135,"lng":96.749399},
                        {"nama":"Sumatera Utara","lat":2.115354,"lng":99.545097},
                        {"nama":"Riau","lat":0.293347,"lng":101.706829},
                        {"nama":"Kepulauan Riau","lat":3.945651,"lng":108.142867},
                        {"nama":"Jakarta","lat":-6.208763,"lng":106.845599},
                        {"nama":"Jawa Barat","lat":-6.914744,"lng":107.609810},
                        {"nama":"Jawa Timur","lat":-7.245972,"lng":112.737991},
                        {"nama":"Bali","lat":-8.409518,"lng":115.188919},
                        {"nama":"Kalimantan Barat","lat":-0.278781,"lng":111.475285},
                        {"nama":"Kalimantan Timur","lat":-0.502106,"lng":117.153709},
                        {"nama":"Sulawesi Utara","lat":1.474830,"lng":124.842079},
                        {"nama":"Sulawesi Selatan","lat":-5.147665,"lng":119.432731},
                        {"nama":"Sulawesi Tenggara","lat":-3.549121,"lng":121.727539},
                        {"nama":"Maluku","lat":-3.238462,"lng":130.145273},
                        {"nama":"Papua Barat","lat":-1.336115,"lng":133.174716},
                        {"nama":"Papua","lat":-2.533333,"lng":140.716667},
                    ],
                },
            }

        return None

    species = get_species(sp)
    if not species:
        st.error("Spesies tidak ditemukan.")
    else:
        # ------- CSS khusus halaman (sama seperti sebelumnya) -------
        SPECIES_CSS = """
        <style>
        .sp-wrap{max-width:1200px;margin:10px auto 24px; padding:0 8px;}
        .sp-bc{background:#eef3ea;color:#3C4030;border-radius:8px;padding:.6rem 1rem;margin:8px 0 14px;}
        .sp-bc a{color:#365902;text-decoration:none}
        .sp-title{color:#2C4001;margin:6px 0 4px;font-size:32px;font-weight:800}
        .sp-sub{color:#3C4030;margin:0 0 16px}
        .sp-tag{display:inline-flex;align-items:center;gap:8px}
        .sp-ic{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;background:#365902;color:#fff;font-size:12px}

        .sp-card{background:#fff;border:1px solid rgba(60,64,48,.18);border-radius:12px;
                 box-shadow:0 4px 12px rgba(44,64,1,.08);}
        .sp-card .head{padding:12px 16px;border-bottom:1px solid rgba(60,64,48,.08);display:flex;gap:8px;align-items:center}
        .sp-card .head h3{margin:0;color:#2C4001}
        .sp-card .body{padding:14px 16px;color:#3C4030}

        .sp-list{list-style:none;padding:0;margin:0}
        .sp-list li{padding:.4rem 0; border-bottom:1px dashed #e6eadf}
        .sp-list li:last-child{border-bottom:none}
        .ok{display:inline-flex;align-items:center;gap:8px}
        .ok:before{content:"";width:18px;height:18px;border-radius:50%;
                   background:#365902;display:inline-block;mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path fill="%23000" d="M13.485 1.929a1 1 0 010 1.414l-7.07 7.071-3.182-3.182a1 1 0 011.414-1.415l1.768 1.768 5.657-5.656a1 1 0 011.414 0z"/></svg>') center/contain no-repeat}

        .sp-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px 18px;margin-top:10px}
        @media (max-width:900px){.sp-grid{grid-template-columns:repeat(2,1fr)}}
        @media (max-width:600px){.sp-grid{grid-template-columns:1fr}}

        .sp-section{margin-top:18px}
        .sp-section .cap{display:flex;align-items:center;gap:8px;color:#2C4001;font-weight:800;font-size:20px;margin:0 0 8px}
        .sp-benefit .head{background:#365902;color:#fff;border-radius:12px 12px 0 0;border-bottom:none}
        .sp-benefit .body{padding:16px}
        </style>
        """
        st.markdown(SPECIES_CSS, unsafe_allow_html=True)

        # ------- Breadcrumb -------
        st.markdown(
            f"""<div class="sp-wrap">
            <nav class="sp-bc" aria-label="breadcrumb">
              <a href="?page=about" target="_self">Tentang Mangrove</a> &nbsp;›&nbsp; {species["nama_latin"]}
            </nav>
            """,
            unsafe_allow_html=True
        )

        # ------- Judul + Nama Lokal -------
        st.markdown(
            f"""<div class="sp-title">{species["nama_latin"]}</div>
            <div class="sp-sub sp-tag"><span class="sp-ic">🏷</span>
              <span>Nama setempat: {", ".join(species["nama_lokal"])}</span>
            </div>""",
            unsafe_allow_html=True
        )

        # ------- Dua kartu atas -------
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""<div class="sp-card">
                    <div class="head"><span class="sp-ic">📄</span><h3>Deskripsi Umum</h3></div>
                    <div class="body">
                      <p style="margin-top:0">{species["deskripsi_umum"]}</p>
                      <div class="sp-section">
                        <div class="cap"><span class="sp-ic">✔</span>Karakteristik Khusus:</div>
                        <ul class="sp-list">
                          {''.join(f'<li class="ok">{k}</li>' for k in species.get("karakteristik", []))}
                        </ul>
                      </div>
                    </div>
                   </div>""",
                unsafe_allow_html=True
            )

        with col2:
            daun = species["daun"]
            st.markdown(
                f"""<div class="sp-card">
                    <div class="head"><span class="sp-ic">🌿</span><h3>Ciri-ciri Daun</h3></div>
                    <div class="body">
                      <ul class="sp-list">
                        <li><b>Bentuk:</b> {daun.get("bentuk","-")}</li>
                        <li><b>Ukuran:</b> {daun.get("ukuran","-")}</li>
                        <li><b>Warna:</b> {daun.get("warna","-")}</li>
                        <li><b>Gagang daun:</b> {daun.get("gagang","-")}</li>
                        <li><b>Letak:</b> {daun.get("letak", daun.get("unit_letak","-"))}</li>
                      </ul>
                      <p style="margin:.5rem 0 0">{daun.get("deskripsi","")}</p>
                    </div>
                   </div>""",
                unsafe_allow_html=True
            )

        # ------- Penyebaran (peta + chip lokasi dalam SATU KOTAK) -------
        loc_list   = species["penyebaran"]["lokasi"]
        coords     = species["penyebaran"]["koordinat"]
        chips_html = "".join(f'<div class="chip">{name}</div>' for name in loc_list)
        coords_json = json.dumps(coords)
        popup_species = species["nama_latin"].replace("'", "\\'")

        st_html(f"""
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <style>
          .card-wrap {{
            border:1px solid #d6decf; border-radius:12px; background:#fff;
            box-shadow:0 2px 10px rgba(0,0,0,.06); padding:14px;
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            color:#3C4030;
          }}
          .title {{ display:flex; align-items:center; gap:8px; color:#2C4001; font-weight:800; margin:4px 0 10px; }}
          #indo-map {{ width:100%; height:430px; border:1px solid #d6decf; border-radius:10px; }}
          .subcap {{ margin:12px 0 6px; font-weight:700; color:#2C4001; display:flex; align-items:center; gap:8px; }}
          .chips {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:8px; }}
          .chip {{
            background:#f5f7f2; border:1px solid #dfe6d8; border-radius:999px;
            padding:6px 10px; font-size:13px; display:flex; align-items:center; gap:6px;
          }}
          .chip::before {{ content:""; width:8px; height:8px; border-radius:50%; background:#365902; display:inline-block; }}
          .marker {{
            background:#365902; width:18px; height:18px; border-radius:50%;
            border:3px solid #fff; box-shadow:0 2px 6px rgba(0,0,0,.25);
          }}
          @media (max-width:900px) {{ .chips{{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
          @media (max-width:600px) {{ .chips{{ grid-template-columns: 1fr; }} }}
        </style>

        <div class="card-wrap">
          <div class="title">🗺️ <h3 style="margin:0">Penyebaran di Indonesia</h3></div>
          <div id="indo-map"></div>
          <div class="subcap">📍 Lokasi Penyebaran</div>
          <div class="chips">{chips_html}</div>
        </div>

        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
          const pts = {coords_json};
          const map = L.map('indo-map', {{ zoomControl: true }}).setView([-2.5489,118.0149], 5);
          L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OpenStreetMap contributors'
          }}).addTo(map);

          const icon = L.divIcon({{ className: 'marker', iconSize: [18,18] }});
          const bounds = [];
          pts.forEach(p => {{
            L.marker([p.lat, p.lng], {{ icon }}).addTo(map)
              .bindPopup('<b style="color:#2C4001;">' + p.nama + '</b><br><small style="color:#3C4030;">{popup_species}</small>');
            bounds.push([p.lat, p.lng]);
          }});
          if (bounds.length) map.fitBounds(bounds, {{ padding: [20,20] }});
        </script>
        """, height=720, scrolling=False)

        # ------- Manfaat (SATU KOTAK, 2 kolom responsif) -------
        benefit_items = ''.join(f'<li class="ok">{m}</li>' for m in species["manfaat"])
        st.markdown(f'''
        <div class="sp-card sp-benefit sp-section">
          <div class="head"><span class="sp-ic">❤️</span><h3>Manfaat</h3></div>
          <div class="body">
            <ul class="sp-list two-col">
              {benefit_items}
            </ul>
          </div>
        </div>
        ''', unsafe_allow_html=True)

        # Footer
        render_footer()
elif page == "faq":
    st.markdown('<div class="faq-wrap">', unsafe_allow_html=True)
    st.markdown('<h1 class="faq-title">Pertanyaan yang Sering Diajukan</h1>', unsafe_allow_html=True)

    faq_html = """
    <div class="accordion" id="faqAccordion">
      <!-- P1 -->
      <div class="ac-item">
        <details open>
          <summary>Apa itu sistem deteksi daun ini?</summary>
          <div class="ac-body">
            Sistem deteksi daun kami dirancang untuk mengidentifikasi dan mengklasifikasikan jenis daun mangrove
            menggunakan teknik pengolahan citra dan machine learning. Aplikasi ini dapat bekerja dalam dua mode:
            <b>Hybrid</b> (fitur manual + CNN) atau <b>CNN-only</b>, sesuai artefak model yang tersedia.
          </div>
        </details>
      </div>

      <!-- P3 -->
      <div class="ac-item">
        <details>
          <summary>Jenis daun apa saja yang bisa diidentifikasi sistem ini?</summary>
          <div class="ac-body">
            Sistem dapat mengidentifikasi 7 spesies: <i>Avicennia marina</i>, <i>Avicennia officinalis</i>,
            <i>Bruguiera gymnorhiza</i>, <i>Heritiera littoralis</i>, <i>Lumnitzera littorea</i>,
            <i>Rhizophora apiculata</i>, dan <i>Sonneratia alba</i>.
            (Daftar kelas aktual juga ditampilkan di sidebar Info Model.)
          </div>
        </details>
      </div>

      <!-- P4 -->
      <div class="ac-item">
        <details>
          <summary>Bagaimana cara menggunakan sistem ini?</summary>
          <div class="ac-body">
            Buka tab <b>Deteksi</b>, unggah satu atau beberapa gambar daun (JPG/PNG/BMP/TIFF),
            lalu klik <b>Prediksi</b>. Sistem akan menghitung fitur, menjalankan model, dan menampilkan
            hasil beserta skor keyakinan. Anda bisa mengaktifkan opsi <i>Masked background</i> dan
            <i>Tampilkan ringkasan fitur</i> untuk visualisasi tambahan.
          </div>
        </details>
      </div>

      <!-- P5 -->
      <div class="ac-item">
        <details>
          <summary>Apakah tersedia aplikasi mobile?</summary>
          <div class="ac-body">
            Saat ini aplikasi berbasis web (Streamlit). Dukungan mobile-native belum tersedia,
            namun antarmuka web responsif sehingga tetap nyaman diakses dari ponsel.
          </div>
        </details>
      </div>

      <!-- P7 -->
      <div class="ac-item">
        <details>
          <summary>Bagaimana jika sistem tidak bisa mengenali daun saya?</summary>
          <div class="ac-body">
            Pastikan gambar fokus, daun tampak utuh, dan pencahayaan memadai. Coba beberapa sudut/ulang foto.
            Jika masih gagal, spesies tersebut mungkin belum ada di basis data pelatihan.
          </div>
        </details>
      </div>

      <!-- Tambahan yang relevan untuk versi Streamlit -->
      <div class="ac-item">
        <details>
          <summary>Apakah gambar saya disimpan di server?</summary>
          <div class="ac-body">
            File yang Anda unggah hanya diproses sementara untuk prediksi. Hasil (fitur) bisa Anda unduh
            sendiri sebagai CSV. Jika perlu kebijakan privasi khusus, sesuaikan pengelolaan file pada folder
            <code>static/uploads</code> atau nonaktifkan penyimpanan permanen.
          </div>
        </details>
      </div>

      <div class="ac-item">
        <details>
          <summary>Bisakah saya mengunduh fitur sebagai CSV?</summary>
          <div class="ac-body">
            Bisa. Setelah prediksi, tombol <b>Download CSV fitur</b> akan muncul. CSV memuat nama file, prediksi,
            confidence, dan kolom fitur manual (sesuai <code>FEATURE_COLS</code> jika mode Hybrid).
          </div>
        </details>
      </div>

      <div class="ac-item">
        <details>
          <summary>Format dan ukuran gambar yang disarankan?</summary>
          <div class="ac-body">
            Gunakan JPG/PNG dengan objek daun jelas dan kontras dari latar. Ukuran bebas; sistem akan
            menyesuaikan ke <b>{w}×{h}</b> piksel (seperti di pengaturan model) saat pra-pemrosesan.
          </div>
        </details>
      </div>
    </div>
    """.format(w=IMG_SIZE[0], h=IMG_SIZE[1])

    st.markdown(faq_html, unsafe_allow_html=True)
    render_footer()
    
# Sidebar “Info Model” tetap ada
with st.sidebar.expander("ℹ️ Info Model", expanded=False):
    st.write(f"**Hybrid:** {IS_HYBRID}")
    st.write(f"**Jumlah input model:** {n_inputs}")
    st.write(f"**Ukuran gambar:** {IMG_SIZE}")
    st.write(f"**Kolom fitur manual (FEATURE_COLS):**")
    st.code(", ".join(FEATURE_COLS) if FEATURE_COLS else "-")

