import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Detection API",
    description="Upload an MRI image and get a tumor detection prediction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config – edit these to match YOUR model ──────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH", "model.h5")   # path to your .h5/.keras file
IMG_SIZE     = int(os.getenv("IMG_SIZE", "224"))      # resize target (e.g. 224 for typical CNNs)
CLASS_NAMES  = os.getenv("CLASS_NAMES", "no_tumor,tumor").split(",")
                                                       # update with your actual class names

# ── Load model once at startup ───────────────────────────────────────────────
model: tf.keras.Model | None = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model file not found at '{MODEL_PATH}'. "
              "Place your model file next to main.py or set MODEL_PATH env var.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"   Input shape : {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")


# ── Helpers ──────────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes → normalised NumPy array ready for inference."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1] normalisation
    return np.expand_dims(arr, axis=0)               # add batch dim → (1, H, W, 3)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Brain Tumor Detection API is running 🧠"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES,
        "img_size": IMG_SIZE,
    }


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Upload an MRI image (JPEG / PNG) and receive a prediction.

    Returns:
    - **predicted_class**: the most likely class label
    - **confidence**: probability for the predicted class (0–1)
    - **probabilities**: probability for every class
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Send JPEG or PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

    # Run inference
    preds = model.predict(tensor, verbose=0)[0]          # shape: (num_classes,)

    # Handle binary sigmoid output (single neuron)
    if len(preds) == 1:
        prob_tumor = float(preds[0])
        probabilities = {CLASS_NAMES[0]: round(1 - prob_tumor, 4),
                         CLASS_NAMES[1]: round(prob_tumor, 4)}
        predicted_class = CLASS_NAMES[1] if prob_tumor >= 0.5 else CLASS_NAMES[0]
        confidence = prob_tumor if prob_tumor >= 0.5 else 1 - prob_tumor
    else:
        # Softmax multi-class output
        probabilities = {
            cls: round(float(p), 4) for cls, p in zip(CLASS_NAMES, preds)
        }
        predicted_idx   = int(np.argmax(preds))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = float(preds[predicted_idx])

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "filename": file.filename,
    })
