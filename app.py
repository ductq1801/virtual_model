from fastapi import FastAPI, File, UploadFile,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from model import Vmodel,Segment,Points,Model_gen
from PIL import Image
import base64
from io import BytesIO

device = "cuda"

app = FastAPI()

model = Vmodel()
segment = Segment(model_type="vit_h",device=device)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post("/segment/")
async def img_segment(data:Points):
    #content = await file.read()
    img = Image.open(BytesIO(base64.b64decode(data['base_image'])))
    x_points = list(map(int, data['x_points']))
    y_points = list(map(int, data['y_points']))
    if len(x_points) != len(y_points):
        raise HTTPException(status_code=400, detail="Size of x list and y list not equal")
    points = zip(x_points,y_points)
    img,mask = segment.segment(points=points,img=img)
    buffered = BytesIO()

    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue())

    mask.save(buffered, format="JPEG")
    mask_base64 = base64.b64encode(buffered.getvalue())
    return {'results':img_base64,
            'mask':mask_base64}
@app.post("/model_gen/")
async def predict_image(data:Model_gen):
    img = Image.open(BytesIO(base64.b64decode(data['image_base64'])))
    mask = Image.open(BytesIO(base64.b64decode(data['mask_base64'])))
    user_prompt = data['user_prompt']
    user_negaprompt = data['user_negaprompt']
    quality = data['quality']
    n_sample = data['n_sample']

    output = Vmodel.gen_img(image=img,mask=mask,add_prompt=user_prompt,add_nega_prompt=user_negaprompt,steps=quality,n_samples=n_sample)
    buffered = BytesIO()
    results = []
    for img in output:
        img.save(buffered, format="JPEG")
        results.append(base64.b64encode(buffered.getvalue()))
    return {"results":results}