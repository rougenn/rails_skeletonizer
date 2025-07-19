from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from global_vars import *

# Load image
pa = SUBIMAGE_EXAMPLE
print(pa)
img = cv.imread(pa)

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Threshold to binary
_, binary = cv.threshold(gray, 127, 1, cv.THRESH_BINARY_INV)  
# NOTE: skeletonize expects 0 and 1, so use maxval=1

# Perform skeletonization
skeleton = skeletonize(1 - binary)

# Show results
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

# plt.subplot(1, 3, 2)
# plt.title("Binary")
# plt.imshow(binary, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Skeleton")
plt.imshow(skeleton, cmap='gray')

plt.show()
