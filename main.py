import io, pickle
import numpy as np
from PIL import ImageOps
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

with open("mnist_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("L")
    pil_image = ImageOps.invert(pil_image)
    pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(pil_image)
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, -1)
    prediction = model.predict(img_array)
    return {"prediction": int(prediction[0])}
