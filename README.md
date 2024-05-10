### Setup:
1. Install requirements:
```
pip3 install -r requirements.txt
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
 ```

2. Weights requirements:
download [this](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) then put in virtual_model/checkpoints
### Demo:
Demo with gradio
```
python demo.py
```
###API
1. Clothes segmentation
```
import requests
url = 
"""
input:
    base_image: string base64 image
    x_points: [x1,x2,x3] list x-axis of points
    y_points: [y1,y2,y3] list y-axis of points
output:
    {
        'results' : string base64 image + mask (latin1 encode),
        'mask': string base64 only mask,
    }
"""
mydata = {
        'base_image': img_str,
        'x_points':[720,320],
        'y_points':[350,265],
        }

x = requests.post(url=url+'/segment/',json=mydata)
```
2. Model generate
```
import requests

#api
url = 

"""
input:
    img_base64: string base64 image
    mask_base64: string base64 mask
    user_prompt: string prompt
    negative_prompt: string negative prompt
    quality: int 0-2
    n_sample: int
output:
    {
        'results': list of string base64 image 
    }
"""
mydata = {'img_base64':str_img,
          'mask_base64':str_mask,
          'user_prompt':None,
          'negative_prompt':None,
          'quality':2,
          "n_sample":2,}

x = requests.post(url=url+'/model_gen/',json=mydata)
```
See notebook/sample_input.ipynb for more details