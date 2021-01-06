import matplotlib.pyplot as plt
import cv2
import numpy as np

img1 = cv2.imread("Reef1.png")
img2 = cv2.imread("Reef2.png")
img_son = img2.copy()

img1rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

gray1 = cv2.cvtColor(img1rgb, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2rgb, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(1000)

keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)

search_params=dict(checks=32)

matcher = cv2.FlannBasedMatcher(index_params, search_params)

matches = matcher.knnMatch(descriptors1, descriptors2, 2)

goodMatches = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        goodMatches.append(m)

fig = plt.figure(figsize=(50, 50))
imMatches = cv2.drawMatches(img1rgb, keypoints1, img2rgb, keypoints2, goodMatches, None, flags= 2)

points1 = np.zeros((len(goodMatches), 2), dtype=np.float32)
points2 = np.zeros((len(goodMatches), 2), dtype=np.float32)

for i, match in enumerate(goodMatches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

print(points1)
print(points2)

height, width, channels = img2.shape
warped = cv2.warpPerspective(img1, h, (width, height))
warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


# Show images
ax = fig.add_subplot(2, 2, 1)
ax.set_title('original first image')
plt.imshow(img1rgb)

ax = fig.add_subplot(2, 2, 2)
ax.set_title('Matched features')
plt.imshow(imMatches)

ax = fig.add_subplot(2, 2, 3)
ax.set_title('warped')
plt.imshow(warped_rgb)
plt.show()


########################################################################################

def bitwise_alma(img):
    hsv1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_pembe = np.array([105, 55, 66])
    upper_pembe = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    bitwise1 = cv2.bitwise_and(img, img, mask=mask1)


    lower_beyaz = np.array([90, 14, 178])
    upper_beyaz = np.array([137, 68, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)
    bitwise2 = cv2.bitwise_and(img, img, mask=mask2)

    top_bitwise =cv2.add(bitwise1,bitwise2)
    return top_bitwise

def mask_alma_beyaz(img):
    lower_beyaz = np.array([90, 14, 178])
    upper_beyaz = np.array([137, 68, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)
    return mask2

def mask_alma_pembe(img):
    lower_pembe = np.array([105, 55, 66])
    upper_pembe = np.array([179, 255, 255])

    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    return mask1

########################################################################################

mask_beyaz_img2 = mask_alma_beyaz(img2)
#cv2.imshow("Beyaz Mask 2", mask_beyaz_img2)
mask_beyaz_wr = mask_alma_beyaz(warped)
#cv2.imshow("Beyaz Mask 1", mask_beyaz_it)

xor_beyaz = cv2.bitwise_xor(mask_beyaz_wr, mask_beyaz_img2)
#cv2.imshow("Xor", xor)
xor_beyaz_blur = cv2.medianBlur(xor_beyaz,5)

contours_beyaz, hierarchy = cv2.findContours(xor_beyaz_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull_beyaz = []

fark_beyaz = np.zeros((xor_beyaz.shape[0], xor_beyaz.shape[1], 3), np.uint8)

for i in range(len(contours_beyaz)):
    area_beyaz =cv2.contourArea(contours_beyaz[i])
    if area_beyaz > 100:

        (x,y,w,h) = cv2.boundingRect(contours_beyaz[i])
        cv2.rectangle(img_son,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),1)
        hull_beyaz.append(cv2.convexHull(contours_beyaz[i], False))
    else:
        pass


for i in range(len(contours_beyaz)):
    cv2.drawContours(fark_beyaz, contours_beyaz, i, (255, 255, 255), 1, 8)
for i in range(len(hull_beyaz)):
    cv2.drawContours(fark_beyaz, hull_beyaz, i, (0, 255, 0), 1, 8)


#################################################################################

mask_pembe_img2 = mask_alma_pembe(img2)
#cv2.imshow("Pembe Mask 2",mask_pembe_img2)
mask_pembe_wr = mask_alma_pembe(warped)
#cv2.imshow("Pembe Mask 1",mask_pembe_it)

xor_pembe = cv2.bitwise_xor(mask_pembe_wr, mask_pembe_img2)
#cv2.imshow("Xor", xor)
xor_pembe_blur = cv2.medianBlur(xor_pembe,5)

contours_pembe, hierarchy2 = cv2.findContours(xor_pembe_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull_pembe = []

fark_pembe = np.zeros((xor_pembe.shape[0], xor_pembe.shape[1], 3), np.uint8)

for i in range(len(contours_pembe)):
    area_pembe =cv2.contourArea(contours_pembe[i])
    if area_pembe > 100:

        (x, y, w, h) = cv2.boundingRect(contours_pembe[i])
        cv2.rectangle(img_son,(x-5,y-5),(x+w+5,y+h+5),(0,0,255),1)
        hull_pembe.append(cv2.convexHull(contours_pembe[i], False))
    else:
        pass


for i in range(len(contours_pembe)):
    cv2.drawContours(fark_pembe, contours_pembe, i, (255, 255, 255), 1, 8)
for i in range(len(hull_pembe)):
    cv2.drawContours(fark_pembe, hull_pembe, i, (0, 255, 0), 1, 8)


#################################################################################

cv2.imshow("img_son",img_son)
cv2.waitKey(0)
cv2.destroyAllWindows()