import torch
from diffusers import StableDiffusionInpaintPipeline,StableDiffusionControlNetInpaintPipeline, ControlNetModel,DDIMScheduler
from controlnet_aux import DWposeDetector
from transformers import pipeline
import cv2
import numpy as np
from PIL import Image
import PIL
import random
class vmodel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        multi_controlnet = [
            ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16,low_cpu_mem_usage=True
        ),
            ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16,low_cpu_mem_usage=True), # for shuffle
            ControlNetModel.from_pretrained("frankjoshua/control_v11f1p_sd15_depth", torch_dtype=torch.float16,low_cpu_mem_usage=True),
        ]
        self.sd_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "stablediffusionapi/epicrealismnew",controlnet=multi_controlnet ,torch_dtype=torch.float16,low_cpu_mem_usage=True
        ).to(self.device)
        self.sd_pipe.enable_xformers_memory_efficient_attention()

        det_config= '/content/controlnet_aux/src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
        det_ckpt= 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
        pose_config= '/content/controlnet_aux/src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py'
        pose_ckpt= 'https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth'

        self.dwpose = DWposeDetector(det_config=det_config, det_ckpt=det_ckpt, pose_config=pose_config, pose_ckpt=pose_ckpt, device=device)
        self.depth_estimator = pipeline('depth-estimation')
        self.base_prompt = "human fashion model, (FKAA, TXAA, SSAO:1.3), in forest, sun, shadow, Photography, (realistic:1.3), (masterpiece:1.2), (photorealistic:1.4), (best quality), (detailed skin:1.2), (detailed clonths:1.2)"
        self.base_negative_prompt = "(extra clothes:1.3), cartoon, (aliasing:1.3), (worst quality, low quality), (deformed, distorted, disfigured, bad eyes), wrong nose, weird mouth, strange ears, bad anatomy, wrong anatomy, bad eyes, amputation, extra limb, missing limb, extra toes, missing toe, (bad teeth, mutated hands, wrong fingers:1.2), disconnected limbs, mutation, ugly, disgusting, (bad_pictures:1.2), (different_skin_color:1.4)"
        self.generator = torch.Generator(self.device).manual_seed(random.randint(0, 100000))
    def new_generator(self):
        self.generator = torch.Generator(self.device).manual_seed(random.randint(0, 100000))
    def get_canny(self,img:np.ndarray) -> PIL.Image:
        canny_image = cv2.Canny(img, 60, 200)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        return Image.fromarray(canny_image)
    
    def gen_img(self,image:PIL.Image,mask:PIL.Image,add_prompt=None,add_nega_prompt=None) -> PIL.Image:
        w, h = image.size
        control1 = self.dwpose(image).resize((w, h))
        mask = self.sd_pipe.mask_processor.blur(mask, blur_factor=14)
        img_np = np.array(image)
        control2 = self.get_canny(img_np)
        control3 = self.depth_estimator(image)['depth']
        prompt = self.base_prompt
        negative_prompt = self.base_negative_prompt
        if add_prompt:
            prompt = ','.join(add_prompt,prompt)
        if add_nega_prompt:
            negative_prompt = ','.join(add_prompt,negative_prompt)
        result = self.sd_pipe(guidance_scale=9.0,num_inference_steps=60,image=image ,\
                              mask_image=mask,control_image = [control1,control2,control3],\
                                controlnet_conditioning_scale=[0.85,,0.55,0.55], prompt=prompt, \
                                    negative_prompt=negative_prompt, generator=generator)
        result = result.images[0]
        return result
        
