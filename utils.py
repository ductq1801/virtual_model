import PIL
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import random
def PIL_to_base64(image:PIL.Image):
    try:
        # Convert NumPy array to PIL Image
        

        # Convert PIL Image to binary data
        with BytesIO() as buffer:
            # You can choose a different format if needed
            image.save(buffer, format="PNG")
            binary_data = buffer.getvalue()

        # Encode binary data to base64
        base64_encoded = base64.b64encode(binary_data).decode("utf-8")

        return base64_encoded
    except Exception as e:
        print(f"Error: {e}")
        return None
def base64_to_PIL(img_str:str):
    return Image.open(BytesIO(base64.b64decode(img_str.encode('utf-8'))))
def ranc(img):
  color = random.randint(0,255) 
  img = img*color
  return color,img
def wh2whc(img):
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        r,rimg = ranc(img)
        g,gimg = ranc(img)
        b,bimg = ranc(img)
        im = np.concatenate((rimg,gimg,bimg),axis=2)
        img = Image.fromarray(im.astype(np.uint8))
        return '{},{},{}'.format(r,g,b),img
    elif len(img.shape) == 3 and img.shape[2] == 1:
        r,rimg = ranc(img)
        g,gimg = ranc(img)
        b,bimg = ranc(img)
        im = np.concatenate((rimg,gimg,bimg),axis=2)
        img = Image.fromarray(im.astype(np.uint8))
        return '{},{},{}'.format(r,g,b),img
    else:
        raise TypeError("img dim should in [2,3]")
def dwpose_padd(img,dwpose,pad):
  w,h = img.size
  h = h + pad
  p_img = Image.new(img.mode, (w, h), (0, 0, 0))
  p_img.paste(img, (0, pad))
  p_pose = dwpose(p_img)
  p_pose = p_pose.resize((p_img.size))
  res = Image.new(p_pose.mode, (w, h-pad), (0, 0, 0))
  res.paste(p_pose, (0, -pad))
  return res

def np_to_base64(img):
   im_bytes = img.tobytes()
   im_b64 = base64.b64encode(im_bytes)
   return im_b64
def mask_2_transmask(img):
  im = np.array(img)
  w,h,c = im.shape
  tl = np.zeros((w,h,4))
  tl[:,:,:3] = img
  tl[:,:,3] = 0
  tl[:,:,3][im[:,:,0]>0] = 255
  tl[:,:,3][im[:,:,1]>0] = 255
  tl[:,:,3][im[:,:,2]>0] = 255
  return Image.fromarray(tl.astype(np.uint8))
def encode_np_array_to_base64(image_array):
    try:
        # Convert NumPy array to PIL Image
        image = Image.fromarray(image_array)

        # Convert PIL Image to binary data
        with BytesIO() as buffer:
            # You can choose a different format if needed
            image.save(buffer, format="PNG")
            binary_data = buffer.getvalue()

        # Encode binary data to base64
        base64_encoded = base64.b64encode(binary_data).decode("utf-8")

        return base64_encoded
    except Exception as e:
        print(f"Error: {e}")
        return None
