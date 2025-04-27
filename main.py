from simple_lama_inpainting import SimpleLama
from PIL import Image
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# segment and get mask
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)

# 创建预测器
predictor = SamPredictor(sam)

image = cv2.imread('./image/dog.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)

input_point = np.array([[400, 500]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

highest_score_mask = masks[np.argmax(scores)]

mask = highest_score_mask.astype(np.uint8) * 255
# mask = cv2.COLOR_BGR2GRAY(mask)

image = cv2.resize(image,[256, 256])
mask = cv2.resize(mask,[256,256])

# lama for inpaint
simple_lama = SimpleLama()

img_path = r"./tests/data/img.png"
mask_path = r"./tests/data/mask.png"

# image = Image.open(img_path)
# mask = Image.open(mask_path).convert('L')

result = simple_lama(image, mask)
result.save("inpainted.png")
