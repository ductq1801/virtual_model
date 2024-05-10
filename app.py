from fastapi import FastAPI, File, UploadFile,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from model import Vmodel,Segment,Points,Model_gen
from PIL import Image
import base64
from io import BytesIO
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import numpy as np
from utils import PIL_to_base64,base64_to_PIL
device = "cuda"

app = FastAPI()

model = Vmodel()
segment = Segment(model_type="vit_h",device=device)
aut = "2gH5CZSLKRoH536OP1RGMXBq0nX_7A8G3sfXEDSJsDJ4jCHpo"
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
@app.get("/")
def home():
  return "hello"
@app.post("/segment/")
async def img_segment(data:Points):
    img = Image.open(BytesIO(data.base_image.encode('latin1')))
    img_np = np.array(img)
    if len(data.x_points) != len(data.y_points):
        raise HTTPException(status_code=400, detail="Size of x list and y list not equal")
    points = zip(data.y_points,data.x_points)
    img,mask = segment.points_segment(points_in=points,img=img_np)
    buffered1 = BytesIO()
    buffered2 = BytesIO()
    img.save(buffered1, format="JPEG")
    img_base64 = buffered1.getvalue().decode('latin1')

    mask.save(buffered2, format="JPEG")
    mask_base64 = buffered2.getvalue().decode('latin1')
    return {'results':img_base64,
            'mask':mask_base64}
@app.post("/model_gen/")
async def predict_image(data:Model_gen):
    img = base64_to_PIL(data.img_base64)
    mask = base64_to_PIL(data.mask_base64)
    user_prompt = data.user_prompt
    user_negaprompt = data.user_negaprompt
    quality = data.quality
    step = max(0,quality)*20 + 40
    if quality==0:
      step = 40
    elif quality==1:
      step = 60
    else:
      step = 80
    n_sample = data.n_sample
    output = model.gen_img(image=img,mask=mask,add_prompt=user_prompt,add_nega_prompt=user_negaprompt,steps=step,n_samples=n_sample)
    results = []
    print(output)
    for im in output:
        results.append(PIL_to_base64(im))
    return {"results":results}
if __name__ == "__main__":
  ngrok.set_auth_token(aut)
  ngrok_tunnel = ngrok.connect(8000)
  
  print('Public URL:', ngrok_tunnel.public_url)
  nest_asyncio.apply()
  uvicorn.run(app, port=8000)