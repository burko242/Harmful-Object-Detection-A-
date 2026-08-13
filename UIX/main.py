from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI(title="harmful_object_detect_API")

model= YOLO("models/harmful_objects.pt")

@app.get("/")
def ana_sayfa():
    return{"mesaj": "YOLO API sistemine hoşgeldiniz. Sistem aktif."}
@app.post("/predict/")
async def resim_tahmin_et(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    sonuclar = model(img)
    tespitler = []  
    base64_resim = ""

    for sonuc in sonuclar:
        kutular = sonuc.boxes
        for kutu in kutular:
            nesne_id = int(kutu.cls[0])
            nesne_adi = model.names[nesne_id]
            guven_skoru = round(float(kutu.conf[0]) * 100, 2)

            tespitler.append({
                "nesne" : nesne_adi,
                "guven" : guven_skoru
            })

        cizilmis_resim = sonuc.plot()
        _, buffer = cv2.imencode('.jpg', cizilmis_resim)
        base64_resim = base64.b64encode(buffer).decode('utf-8')

    return {
    "orijinal_dosya_adi": file.filename,
    "tespitler": tespitler,
    "isaretli_resim_base64": base64_resim  
}