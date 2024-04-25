
from PIL import Image
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