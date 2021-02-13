import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import itertools

img1 = cv2.imread("1.1f.png")
img2 = cv2.imread("3.1.png")

#img2 = cv2.resize(img2,(img1.shape[1],img1.shape[0]))

copyimg1 = img1.copy()
copyimg2 = img2.copy()

kernel = np.ones((5,5),np.uint8)

def bitwise_alma(img):
    hsv1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_pembe = np.array([148, 0, 100])
    upper_pembe = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    bitwise1 = cv2.bitwise_and(img, img, mask=mask1)


    lower_beyaz = np.array([63, 28, 183])
    upper_beyaz = np.array([81, 255, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)
    bitwise2 = cv2.bitwise_and(img, img, mask=mask2)

    top_bitwise =cv2.add(bitwise1,bitwise2)
    return top_bitwise

def mask_alma(img):
    hsv1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_pembe = np.array([148, 0, 100])
    upper_pembe = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)

    lower_beyaz = np.array([30, 0, 180])
    upper_beyaz = np.array([85, 100, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)

    top_mask =cv2.add(mask1,mask2)
    return top_mask

def mask_alma_beyaz(img):
    lower_beyaz = np.array([30, 0, 180])
    upper_beyaz = np.array([85, 100, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)
    return mask2

def mask_alma_sari(img):
    lower_sari = np.array([20, 0, 0])
    upper_sari = np.array([50, 255, 255])

    hsvSari = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maskSari = cv2.inRange(hsvSari, lower_sari, upper_sari)
    return maskSari

def mask_alma_kirmizi_yesil(img):
    lower_renk = np.array([0, 255, 255])
    upper_renk = np.array([70, 255, 255])

    hsvKS = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maskKS = cv2.inRange(hsvKS, lower_renk, upper_renk)
    return maskKS

def mask_alma_pembe(img):
    lower_pembe = np.array([148, 0, 100])
    upper_pembe = np.array([179, 255, 255])

    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    return mask1

def bit_xor(img1,img2,imgson):
    img1 = cv2.erode(img1, np.ones((3, 3)), iterations=3)
    img2 = cv2.erode(img2, np.ones((3, 3)), iterations=3)
    xor = cv2.bitwise_xor(img1, img2)
    #cv2.imshow("Xor", xor)
    xor_blur = cv2.medianBlur(xor,5)
    #cv2.imshow("Blur Xor",xor)

    contours, hierarchy = cv2.findContours(xor_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull = []

    fark = np.zeros((xor.shape[0], xor.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        area =cv2.contourArea(contours[i])
        print(area)
        if area < 5000:
            print("Çıkarıldı")
        else:
            print("Eklendi")
            (x,y,w,h) = cv2.boundingRect(contours[i])
            cv2.rectangle(imgson,(x-5,y-5),(x+w+5,y+h+5),(0,0,255),1)
            hull.append(cv2.convexHull(contours[i], False))



    for i in range(len(contours)):
        cv2.drawContours(fark, contours, i, (255, 255, 255), 1, 8)
    for i in range(len(hull)):
        cv2.drawContours(fark, hull, i, (0, 255, 0), 1, 8)
    return (xor,xor_blur,fark)

#######################################################################################################################

def generate_nonadjacent_combination(input_list, take_n):
    """
    It generates combinations of m taken n at a time where there is no adjacent n.
    INPUT:
        input_list = (iterable) List of elements you want to extract the combination
        take_n =     (integer) Number of elements that you are going to take at a time in
                     each combination
    OUTPUT:
        all_comb =   (np.array) with all the combinations
    """
    all_comb = []
    for comb in itertools.combinations(input_list, take_n):
        comb = np.array(comb)
        d = np.diff(comb)
        fd = np.diff(np.flip(comb))
        if len(d[d == 1]) == 0 and comb[-1] - comb[0] != 7:
            all_comb.append(comb)
            #print(comb)
    return all_comb

def populate_intersection_kernel(combinations):
    """
    Maps the numbers from 0-7 into the 8 pixels surrounding the center pixel in
    a 9 x 9 matrix clockwisely i.e. up_pixel = 0, right_pixel = 2, etc. And
    generates a kernel that represents a line intersection, where the center
    pixel is occupied and 3 or 4 pixels of the border are ocuppied too.
    INPUT:
        combinations = (np.array) matrix where every row is a vector of combinations
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    n = len(combinations[0])
    template = np.array((
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, -1, -1]), dtype="int")
    match = [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)]
    kernels = []
    for n in combinations:
        tmp = np.copy(template)
        for m in n:
            tmp[match[m][0], match[m][1]] = 1
        kernels.append(tmp)
    return kernels

def give_intersection_kernels():
    """
    Generates all the intersection kernels in a 9x9 matrix.
    INPUT:
        None
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    input_list = np.arange(8)
    taken_n = [4, 3]
    kernels = []
    for taken in taken_n:
        comb = generate_nonadjacent_combination(input_list, taken)
        tmp_ker = populate_intersection_kernel(comb)
        kernels.extend(tmp_ker)
    return kernels

def find_line_intersection(input_image, show=0):
    """
    Applies morphologyEx with parameter HitsMiss to look for all the curve
    intersection kernels generated with give_intersection_kernels() function.
    INPUT:
        input_image =  (np.array dtype=np.uint8) binarized m x n image matrix
    OUTPUT:
        output_image = (np.array dtype=np.uint8) image where the nonzero pixels
                       are the line intersection.
    """
    kernel = np.array(give_intersection_kernels())
    output_image = np.zeros(input_image.shape)
    for i in np.arange(len(kernel)):
        out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel[i, :, :])
        out = cv2.dilate(out,np.ones((5,5),np.uint8),iterations= 3)
        output_image = output_image + out
    if show == 1:
        show_image = np.reshape(np.repeat(input_image, 3, axis=1),
                                (input_image.shape[0], input_image.shape[1], 3)) * 255
        show_image[:, :, 1] = show_image[:, :, 1] - output_image * 255
        show_image[:, :, 2] = show_image[:, :, 2] - output_image * 255
        cv2.imshow("line transition",show_image)
        plt.imshow(show_image)
    return output_image

def find_endoflines(input_image, show=0):
    """
    """
    kernel_0 = np.array((
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, 1, -1]), dtype="int")

    kernel_1 = np.array((
        [-1, -1, -1],
        [-1, 1, -1],
        [1, -1, -1]), dtype="int")

    kernel_2 = np.array((
        [-1, -1, -1],
        [1, 1, -1],
        [-1, -1, -1]), dtype="int")

    kernel_3 = np.array((
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, -1]), dtype="int")

    kernel_4 = np.array((
        [-1, 1, -1],
        [-1, 1, -1],
        [-1, -1, -1]), dtype="int")

    kernel_5 = np.array((
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, -1, -1]), dtype="int")

    kernel_6 = np.array((
        [-1, -1, -1],
        [-1, 1, 1],
        [-1, -1, -1]), dtype="int")

    kernel_7 = np.array((
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]), dtype="int")

    kernel = np.array((kernel_0, kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7))
    output_image = np.zeros(input_image.shape)
    for i in np.arange(8):
        out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel[i, :, :])
        out = cv2.dilate(out, np.ones((5, 5), np.uint8), iterations=3)
        output_image = output_image + out

    if show == 1:
        show_image = np.reshape(np.repeat(input_image, 3, axis=1),
                                (input_image.shape[0], input_image.shape[1], 3)) * 255
        show_image[:, :, 1] = show_image[:, :, 1] - output_image * 255
        show_image[:, :, 2] = show_image[:, :, 2] - output_image * 255
        plt.imshow(show_image)

    return output_image  # , np.where(output_image == 1)

#######################################################################################################################

def skel(img):
    mask = mask_alma(img)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    binary = closing > filters.threshold_otsu(closing)

    skeleton_lee = skeletonize(binary, method='lee')

    lint_img = find_line_intersection(skeleton_lee, 0)

    img1 = np.uint8(lint_img)

    contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    points = []

    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        points.append(((int(x + (w) / 2), int(y + (h) / 2))))
        print(((int(x + (w) / 2), int(y + (h) / 2))))
    print("########")
    return (skeleton_lee,lint_img,points)


iskelet1 , kesisim1 , points1 = skel(img1)
iskelet2 , kesisim2 , points2 = skel(img2)

cv2.imshow("asdfasdf",find_endoflines(iskelet1))

m = len(points1)
n = len(points2)

if m == n:
    keypoints1 = np.float32(points1)
    keypoints2 = np.float32(points2)
elif m < n:
    keypoints1 = np.float32(points1)
    keypoints2 = np.float32(points2[:m])
else:
    keypoints1 = np.float32(points1[:n])
    keypoints2 = np.float32(points2)

h, mask = cv2.findHomography(keypoints1, keypoints2, cv2.RANSAC)
height, width, channels = img2.shape
warped = cv2.warpPerspective(copyimg1, h, (width, height))

added = cv2.addWeighted(bitwise_alma(warped), 0.5, copyimg2, 0.5, 0)

pWarped = mask_alma_pembe(warped)
pIMG2 = mask_alma_pembe(img2)

sonuc = bit_xor(pWarped,pIMG2,added)
#cv2.imshow("XOR BLUR Pembe", sonuc[1])
#cv2.imshow("SONUC Pembe", sonuc[2])

mWarped = mask_alma_beyaz(warped)
mIMG2 = mask_alma_beyaz(img2)

sonuc2 = bit_xor(mWarped, mIMG2,added)
#cv2.imshow("XOR BLUR Beyaz", sonuc2[1])
#cv2.imshow("SONUC Beyaz", sonuc2[2])

cv2.imshow("Added", added)

for i in range(len(points1)):
    (x, y) = points1[i]
    cv2.circle(img1, (x, y), 3, (255, 0, 0), 1, -1)
    cv2.circle(iskelet1, (x, y), 12, (255, 0, 0), 1, -1)
    if i < 3:
        e = 10
        cv2.line(img1, (x + e, y), (x + e, 0), (0, 0, 255), 1)
        cv2.line(img1, (x - e, y), (x - e, 0), (0, 0, 255), 1)

for i in range(len(points2)):
    (x, y) = points2[i]
    cv2.circle(img2, (x, y), 3, (255, 0, 0), 1, -1)
    cv2.circle(iskelet2, (x, y), 12, (255, 0, 0), 1, -1)
    if i < 3:
        e = 10
        cv2.line(img2, (x + e, y), (x + e, 0), (0, 0, 255), 1)
        cv2.line(img2, (x - e, y), (x - e, 0), (0, 0, 255), 1)

cv2.imshow("Skel 1",iskelet1)
cv2.imshow("Skel 2",iskelet2)

cv2.imshow("IMG 1",img1)
cv2.imshow("IMG 2",img2)

cv2.waitKey(0)
cv2.destroyAllWindows()