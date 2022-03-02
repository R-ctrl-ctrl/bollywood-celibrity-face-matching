import cv2
from PIL import Image

sample_img = cv2.imread('samples/salmon_copy.png')
print(type(sample_img))
img = Image.fromarray(sample_img)

print(type(img))