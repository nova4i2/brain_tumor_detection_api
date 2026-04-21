# 🧠 Brain Tumor Detection API

A FastAPI-based REST API that accepts MRI images and returns brain tumor predictions using your trained Keras model.

---

## 📁 Project Structure

```
brain_tumor_api/
├── main.py            ← FastAPI application
├── requirements.txt   ← Python dependencies
├── render.yaml        ← Render deployment config
├── model.h5           ← YOUR model file (add this!)
└── README.md
```

---

## ⚙️ Configuration

Edit these values in `render.yaml` (or set as environment variables):

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model.h5` | Path to your saved Keras model |
| `IMG_SIZE` | `224` | Image resize dimension (e.g. 224 for VGG/ResNet) |
| `CLASS_NAMES` | `no_tumor,tumor` | Comma-separated class labels (match training order) |

**Example CLASS_NAMES for 4-class model:**
```
no_tumor,glioma,meningioma,pituitary
```

---

## 🚀 Deploy on Render (Free)

### Step 1 – Prepare your repo
```bash
# Create a GitHub repo and push all files including your model
git init
git add .
git commit -m "Initial brain tumor API"
git remote add origin https://github.com/YOUR_USERNAME/brain-tumor-api.git
git push -u origin main
```

> ⚠️ If your model file is > 100 MB, use [Git LFS](https://git-lfs.github.com/):
> ```bash
> git lfs install
> git lfs track "*.h5"
> git add .gitattributes
> git commit -m "Track model with LFS"
> ```

### Step 2 – Deploy on Render
1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` — click **Deploy**
4. Wait ~3–5 minutes for the build

### Step 3 – Test your live API
```bash
curl -X POST https://your-app.onrender.com/predict \
  -F "file=@brain_mri.jpg"
```

---

## 🧪 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your model file
cp /path/to/your/model.h5 .

# 3. Start the server
uvicorn main:app --reload

# 4. Open interactive docs
# http://localhost:8000/docs
```

---

## 📡 API Endpoints

### `GET /`
Health check.

### `GET /health`
Returns model status, class names, and image size config.

### `POST /predict`
Upload an MRI image → get prediction.

**Request:** `multipart/form-data` with field `file` (JPEG or PNG)

**Response:**
```json
{
  "predicted_class": "tumor",
  "confidence": 0.9821,
  "probabilities": {
    "no_tumor": 0.0179,
    "tumor": 0.9821
  },
  "filename": "mri_scan.jpg"
}
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Model is not loaded` | Check `MODEL_PATH` env var and that the file exists |
| Wrong predictions | Verify `IMG_SIZE` and `CLASS_NAMES` match training config |
| 503 on Render free tier | Free tier sleeps after 15 min inactivity — first request wakes it up |
| Model > 100MB won't push | Use Git LFS (see Step 1) |
