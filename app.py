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
from utils import PIL_to_base64,base64_to_PIL,encode_np_array_to_base64,wh2whc
from ultralytics.models.fastsam import FastSAM, FastSAMPrompt

device = "cuda"

app = FastAPI()

model = Vmodel()
#segment = Segment(model_type="vit_h",device=device)
fast_segment = FastSAM('checkpoints/FastSAM.pt')

aut = "2gH5CZSLKRoH536OP1RGMXBq0nX_7A8G3sfXEDSJsDJ4jCHpo"
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
@app.post("/segment/")
async def img_segment(data:Points):
    img = base64_to_PIL(data.base_image)
    #img_np = np.array(img)
    everything_results = fast_segment(img, device=device, retina_masks=True, conf=0.4, iou=0.9)
    prompt_process  = FastSAMPrompt(img, everything_results, device=device)
    ann = prompt_process.everything_prompt()
    mask_base64 = []
    for mask in ann[0]:
        msk = mask.masks.data.cpu().numpy().squeeze()
        msk = wh2whc(msk)
        mask_base64.append(PIL_to_base64(msk))
    return {'results':mask_base64}
@app.post("/model_gen/")
async def predict_image(data:Model_gen):
    img = base64_to_PIL(data.img_base64)
    mask = np.zeros(img.size())
    for mask in data.mask_base64:
      mask += base64_to_PIL(mask)
    user_prompt = data.user_prompt
    user_negaprompt = data.user_negaprompt
    quality = data.quality
    step = max(0,quality)*20 + 40
    n_sample = data.n_sample
    output = model.gen_img(image=img,mask=mask,add_prompt=user_prompt,add_nega_prompt=user_negaprompt,steps=step,n_samples=n_sample)
    results = []
    for im in output:
        results.append(PIL_to_base64(im))
    return {"results":results}
if __name__ == "__main__":
  ngrok.set_auth_token(aut)
  ngrok_tunnel = ngrok.connect(8000)
  
  print('Public URL:', ngrok_tunnel.public_url)
  nest_asyncio.apply()
  uvicorn.run(app, port=8000)
