
from PIL import Image
from io import BytesIO
def PIL_to_base64(image:PIL.Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str =buffered.getvalue().decode('latin1')
    return img_str
def base64_to_PIL(img_str:str):
    return Image.open(BytesIO((img_str.encode('latin1'))))

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