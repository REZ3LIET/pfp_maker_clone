import cv2 as cv
import numpy as np
from rembg import remove

path = "E:\Github\PFP_Maker\pfp_maker_clone\Data\Input\study_sis.jpg"
dim = 720
img = cv.imread(path)
print("image loaded")

h, w, _ = img.shape
dim_2 = min(h, w)//2
img_crop = img[h//2-dim_2:h//2+dim_2, w//2-dim_2:w//2+dim_2]
img_crop = cv.resize(img_crop, (dim, dim), interpolation=cv.INTER_LINEAR)

print("processing...")
fg = remove(img_crop)
fg = remove(fg)
fg = cv.cvtColor(fg, cv.COLOR_BGRA2BGR)

bg_color = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
bg = (np.ones((dim, dim, 3)) * (0, 0, 0)).astype(np.uint8)
cv.circle(bg, (dim//2, dim//2), dim//2, bg_color, -1)

fg_grey = cv.cvtColor(fg, cv.COLOR_BGR2GRAY)
_, fg_mask = cv.threshold(fg_grey, 0, 255, cv.THRESH_BINARY)

base_mask = cv.bitwise_and(bg, bg, mask=fg_mask)
base_mask = cv.cvtColor(base_mask, cv.COLOR_BGR2GRAY)
_, base_mask = cv.threshold(base_mask, 0, 255, cv.THRESH_BINARY)

fg_final = cv.bitwise_and(fg, fg, mask=base_mask)

base_mask_inv = cv.bitwise_not(base_mask)
bg_final = cv.bitwise_and(bg, bg, mask=base_mask_inv)

pic = fg_final + bg_final

print("Done...")

cv.imshow("PIC", pic)
# cv.imshow("original", img)
# cv.imshow("bg", bg)
cv.waitKey(0)
cv.destroyAllWindows()