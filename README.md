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

download for fastSAM:
```
!wget https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt
```
### Demo:
Demo with gradio
```
python demo.py
```
### API
1. Clothes segmentation
```
import requests

url = 

"""
input:
    base_image: string base64 image
output:
    {
        results : list of string base64 mask,
    }
"""
mydata = {
        'base_image': img_str,
        }

x = requests.post(url=url+'/segment/',json=mydata)
```
2. Model generate
```
import requests

url = 

"""
input:
    img_base64: string base64 image
    mask_base64: string base64 mask
    user_prompt: string prompt
    positive_prompt: string negative prompt
    negative_prompt: string negative prompt
    quality:[optional] int 0-4 | default=1
    n_sample:[optional] int | default=2
output:
    {
        'results': list of string base64 image 
    }
"""
mydata = {img_base64:str_img,
          mask_base64:str_mask,
          user_prompt:None,
          negative_prompt:None,
          quality:2,
          n_sample:2,}

x = requests.post(url=url+'/model_gen/',json=mydata)
```
See notebook/sample_input.ipynb for more details
