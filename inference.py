import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import gc

models = {
	'vit_b': './checkpoints/sam_vit_b_01ec64.pth',
	'vit_l': './checkpoints/sam_vit_l_0b3195.pth',
	'vit_h': './checkpoints/sam_vit_h_4b8939.pth'
}

image_examples = [
    [os.path.join(os.path.dirname(__file__), "./images/53960-scaled.jpg"), 0, []],
    [os.path.join(os.path.dirname(__file__), "./images/2388455-scaled.jpg"), 1, []],
    [os.path.join(os.path.dirname(__file__), "./images/1.jpg"),2,[]],
    [os.path.join(os.path.dirname(__file__), "./images/2.jpg"),3,[]],
    [os.path.join(os.path.dirname(__file__), "./images/3.jpg"),4,[]],
    [os.path.join(os.path.dirname(__file__), "./images/4.jpg"),5,[]],
    [os.path.join(os.path.dirname(__file__), "./images/5.jpg"),6,[]],
    [os.path.join(os.path.dirname(__file__), "./images/6.jpg"),7,[]],
    [os.path.join(os.path.dirname(__file__), "./images/7.jpg"),8,[]],
    [os.path.join(os.path.dirname(__file__), "./images/8.jpg"),9,[]]
]


def plot_boxes(img, boxes):
	img_pil = Image.fromarray(np.uint8(img * 255)).convert('RGB')
	draw = ImageDraw.Draw(img_pil)
	for box in boxes:
		color = tuple(np.random.randint(0, 255, size=3).tolist())
		x0, y0, x1, y1 = box
		x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
		draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
	return img_pil


def segment_one(img, mask_generator, seed=None):
	if seed is not None:
		np.random.seed(seed)
	masks = mask_generator.generate(img)
	sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
	mask_all = np.ones((img.shape[0], img.shape[1], 3))
	for ann in sorted_anns:
		m = ann['segmentation']
		color_mask = np.random.random((1, 3)).tolist()[0]
		for i in range(3):
			mask_all[m == True, i] = color_mask[i]
	result = img / 255 * 0.3 + mask_all * 0.7
	return result, mask_all


def generator_inference(device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh,
                        min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh,
                        input_x, progress=gr.Progress()):
	# sam model
	sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
	mask_generator = SamAutomaticMaskGenerator(
		sam,
		points_per_side=points_per_side,
		pred_iou_thresh=pred_iou_thresh,
		stability_score_thresh=stability_score_thresh,
		stability_score_offset=stability_score_offset,
		box_nms_thresh=box_nms_thresh,
		crop_n_layers=crop_n_layers,
		crop_nms_thresh=crop_nms_thresh,
		crop_overlap_ratio=512 / 1500,
		crop_n_points_downscale_factor=1,
		point_grids=None,
		min_mask_region_area=min_mask_region_area,
		output_mode='binary_mask'
	)

	# input is image, type: numpy
	if type(input_x) == np.ndarray:
		result, mask_all = segment_one(input_x, mask_generator)
		return result, mask_all
	elif isinstance(input_x, str):  # input is video, type: path (str)
		cap = cv2.VideoCapture(input_x)  # read video
		frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc('x', '2', '6', '4'), fps, (W, H), isColor=True)
		while True:
			ret, frame = cap.read()  # read a frame
			if ret:
				result, mask_all = segment_one(frame, mask_generator, seed=2023)
				result = (result * 255).astype(np.uint8)
				out.write(result)
			else:
				break
		out.release()
		cap.release()
		return 'output.mp4'


def predictor_inference(device, model_type, input_x, input_text, selected_points, owl_vit_threshold=0.1):
	# sam model
	sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
	predictor = SamPredictor(sam)
	predictor.set_image(input_x)  # Process the image to produce an image embedding

	if input_text != '':
		# split input text
		input_text = [input_text.split(',')]
		print(input_text)
		# OWL-ViT model
		processor = OwlViTProcessor.from_pretrained('./checkpoints/models--google--owlvit-base-patch32')
		owlvit_model = OwlViTForObjectDetection.from_pretrained("./checkpoints/models--google--owlvit-base-patch32").to(device)
		# get outputs
		input_text = processor(text=input_text, images=input_x, return_tensors="pt").to(device)
		outputs = owlvit_model(**input_text)
		target_size = torch.Tensor([input_x.shape[:2]]).to(device)
		results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_size,
		                                                  threshold=owl_vit_threshold)

		# get the box with best score
		scores = torch.sigmoid(outputs.logits)
		# best_scores, best_idxs = torch.topk(scores, k=1, dim=1)
		# best_idxs = best_idxs.squeeze(1).tolist()

		i = 0  # Retrieve predictions for the first image for the corresponding text queries
		boxes_tensor = results[i]["boxes"]  # [best_idxs]
		boxes = boxes_tensor.cpu().detach().numpy()
		# boxes = boxes[np.newaxis, :, :]
		transformed_boxes = predictor.transform.apply_boxes_torch(torch.Tensor(boxes).to(device),
		                                                          input_x.shape[:2])  # apply transform to original boxes
		# transformed_boxes = transformed_boxes.unsqueeze(0)
		print(transformed_boxes.size(), boxes.shape)
	else:
		transformed_boxes = None

	# points
	if len(selected_points) != 0:
		points = torch.Tensor([p for p, _ in selected_points]).to(device).unsqueeze(1)
		labels = torch.Tensor([int(l) for _, l in selected_points]).to(device).unsqueeze(1)
		transformed_points = predictor.transform.apply_coords_torch(points, input_x.shape[:2])
		print(points.size(), transformed_points.size(), labels.size(), input_x.shape, points)
	else:
		transformed_points, labels = None, None

	# predict segmentation according to the boxes
	masks, scores, logits = predictor.predict_torch(
		point_coords=transformed_points,
		point_labels=labels,
		boxes=transformed_boxes,  # only one box
		multimask_output=False,
	)
	masks = masks.cpu().detach().numpy()
	mask_all = np.ones((input_x.shape[0], input_x.shape[1], 3))
	for ann in masks:
		color_mask = np.random.random((1, 3)).tolist()[0]
		for i in range(3):
			mask_all[ann[0] == True, i] = color_mask[i]
	img = input_x / 255 * 0.3 + mask_all * 0.7
	if input_text != '':
		img = plot_boxes(img, boxes_tensor)  # image + mask + boxes

	# free the memory
	if input_text != '':
		owlvit_model.cpu()
		del owlvit_model
	del input_text
	gc.collect()
	torch.cuda.empty_cache()

	return img, mask_all


def run_inference(device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area,
                  stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh, owl_vit_threshold, input_x,
                  input_text, selected_points=[]):
	# if input_x is int, the image is selected from examples
	if isinstance(input_x, int):
		input_x = cv2.imread(image_examples[input_x][0])
		input_x = cv2.cvtColor(input_x, cv2.COLOR_BGR2RGB)
	if (input_text != '' and not isinstance(input_x, str)) or len(selected_points) != 0:  # user input text or points
		print('use predictor_inference')
		print('prompt text: ', input_text)
		print('prompt points length: ', len(selected_points))
		return predictor_inference(device, model_type, input_x, input_text, selected_points, owl_vit_threshold)
	else:
		print('use generator_inference')
		return generator_inference(device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh,
		                           min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers,
		                           crop_nms_thresh, input_x)
def gen_model(vmodel,img,mask):
  vmodel.new_generator()
  img = gen_model(img,mask)
  return img
