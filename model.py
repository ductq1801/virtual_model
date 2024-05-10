import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from controlnet_aux import DWposeDetector
from transformers import pipeline
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import numpy as np
from PIL import Image
import PIL
import random
from pydantic import BaseModel
from typing import List
models = {
	'vit_b': './checkpoints/sam_vit_b_01ec64.pth',
	'vit_l': './checkpoints/sam_vit_l_0b3195.pth',
	'vit_h': './checkpoints/sam_vit_h_4b8939.pth'
}

colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

class Vmodel:
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

        det_config= './dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
        det_ckpt= 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
        pose_config= './dwpose/dwpose_config/dwpose-l_384x288.py'
        pose_ckpt= 'https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth'

        self.dwpose = DWposeDetector(det_config=det_config, det_ckpt=det_ckpt, \
        pose_config=pose_config, pose_ckpt=pose_ckpt, device=self.device)
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
    def create_mask(self,mask):
        thresh = 200
        fn = lambda x : 255 if x > thresh else 0
        r = mask.convert('L').point(fn, mode='1')
        return r
    def gen_img(self,image:PIL.Image,mask:PIL.Image,add_prompt=None,add_nega_prompt=None,steps=40,n_samples=1) -> PIL.Image:
        image = Image.fromarray(image)
        w, h = image.size
        control1 = self.dwpose(image).resize((w, h))
        mask= Image.fromarray(mask)
        mask = self.create_mask(mask)
        mask = mask.convert('RGB')
        mask = self.sd_pipe.mask_processor.blur(mask, blur_factor=12,)
        img_np = np.array(image)
        control2 = self.get_canny(img_np)
        control3 = self.depth_estimator(image)['depth']
        prompt = self.base_prompt
        negative_prompt = self.base_negative_prompt
        if add_prompt:
            prompt = ','.join(add_prompt,prompt)
        if add_nega_prompt:
            negative_prompt = ','.join(add_prompt,negative_prompt)
        result = self.sd_pipe(guidance_scale=9.0,num_inference_steps=steps,num_images_per_prompt=n_samples,image=image ,\
                              mask_image=mask,control_image = [control1,control2,control3],\
                                controlnet_conditioning_scale=[0.85,0.55,0.55], prompt=prompt, \
                                    negative_prompt=negative_prompt, generator=self.generator)
        result = result
        return result
        

class Segment:
    def __init__(self,model_type,device):
        sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
        self.mask_generator = SamAutomaticMaskGenerator(
		sam,
		crop_overlap_ratio=512 / 1500,
		crop_n_points_downscale_factor=1,
		point_grids=None,
		output_mode='binary_mask'
	)
    def segment_one(self,img:PIL.Image):
        masks = self.mask_generator.generate(img)
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        mask_all = np.ones((img.shape[0], img.shape[1], 3))
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                mask_all[m == True, i] = color_mask[i]
        result = img / 255 * 0.3 + mask_all * 0.7
        return result, mask_all
    def segment(self,img:np.array,points:list[tuple]):
        for point in points:
            cv2.drawMarker(img, point, colors[0], markerType=markers[0], markerSize=10, thickness=3)
        
        if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result, mask_all = self.segment_one(img)
        return Image.fromarray((result * 255).astype(np.uint8)), Image.fromarray((mask_all * 255).astype(np.uint8))
    
class Points(BaseModel):
    x_points : List[int]
    y_points : List[int]
    base_image: str
class Model_gen(BaseModel):
    img_base64:str
    mask_base64:str
    user_prompt:str | None = None
    user_negaprompt:str | None=None
    quality:int | None=0
    n_sample:int | None=1

