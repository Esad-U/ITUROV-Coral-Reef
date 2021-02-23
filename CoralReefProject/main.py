import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import itertools

"""
Üst üste getirme ve tekrar üst üste getirme ve üst derece fark bulma 
"""

cap = cv2.VideoCapture("C:\\Users\\QP\\Desktop\\videos\\video5.mkv")
img1 = cv2.imread("C:\\Users\\QP\\Desktop\\fotolar\\fotoooF2.png")
img1 = cv2.resize(img1,(int(img1.shape[1]/2),int(img1.shape[0]/2)))
copyimg1 = img1.copy()

kernel = np.ones((5,5),np.uint8)

def sortSecond(val):
    return val[1]

def bitwise_alma(img):
    hsv1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_pembe = np.array([105, 0, 68])
    upper_pembe = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    bitwise1 = cv2.bitwise_and(img, img, mask=mask1)


    lower_beyaz = np.array([40, 0, 170])
    upper_beyaz = np.array([90, 95, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)
    bitwise2 = cv2.bitwise_and(img, img, mask=mask2)

    top_bitwise =cv2.add(bitwise1,bitwise2)
    return top_bitwise

def mask_alma(img):
    hsv1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_pembe = np.array([120, 0, 68])
    upper_pembe = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)

    lower_beyaz = np.array([40, 0, 170])
    upper_beyaz = np.array([90, 95, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)

    top_mask =cv2.add(mask1,mask2)
    return top_mask

def mask_alma_beyaz(img):
    lower_beyaz = np.array([40, 0, 170])
    upper_beyaz = np.array([90, 95, 255])

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
    lower_pembe = np.array([120, 0, 68])
    upper_pembe = np.array([179, 255, 255])

    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    return mask1

def bit_xor(img1, img2, imgson, renk, kalinlik):
    #img1 = cv2.erode(img1, np.ones((3, 3)), iterations=3)
    #img2 = cv2.erode(img2, np.ones((3, 3)), iterations=3)
    xor = cv2.bitwise_xor(img1, img2)
    erode = cv2.erode(xor,np.ones((1, 3)))
    closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    dilation = cv2.dilate(opening, np.ones((3, 3), np.uint8), iterations=1)
    xor_blur = cv2.medianBlur(dilation,5)

    contours, hierarchy = cv2.findContours(xor_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull = []

    fark = np.zeros((xor.shape[0], xor.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        #print(area)
        if area < 700:
            #print("Çıkarıldı")
            pass
        else:
            #print("Eklendi")
            (x,y,w,h) = cv2.boundingRect(contours[i])
            cv2.rectangle(imgson,(x-5,y-5),(x+w+5,y+h+5),renk,kalinlik)
            hull.append(cv2.convexHull(contours[i], False))



    for i in range(len(contours)):
        cv2.drawContours(fark, contours, i, (255, 255, 255), 1, 8)
    for i in range(len(hull)):
        cv2.drawContours(fark, hull, i, (0, 255, 0), 1, 8)
    return (xor,xor_blur,fark)

def bit_and(img1, img2, imgson, renk, kalinlik):
    #img1 = cv2.erode(img1, np.ones((3, 3)), iterations=3)
    #img2 = cv2.erode(img2, np.ones((3, 3)), iterations=3)
    bAnd = cv2.bitwise_and(img1, img2)
    erode = cv2.erode(bAnd,np.ones((1, 3)))
    closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    dilation = cv2.dilate(opening, np.ones((3, 3), np.uint8), iterations=1)

    bAnd_blur = cv2.medianBlur(dilation,5)
    #cv2.imshow("Blur Xor",xor)

    contours, hierarchy = cv2.findContours(bAnd_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull = []

    fark = np.zeros((bAnd.shape[0], bAnd.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        #print(area)
        if area < 700:
            #print("Çıkarıldı")
            pass
        else:
            #print("Eklendi")
            (x,y,w,h) = cv2.boundingRect(contours[i])
            cv2.rectangle(imgson,(x-5,y-5),(x+w+5,y+h+5),renk,kalinlik)
            hull.append(cv2.convexHull(contours[i], False))



    for i in range(len(contours)):
        cv2.drawContours(fark, contours, i, (255, 255, 255), 1, 8)
    for i in range(len(hull)):
        cv2.drawContours(fark, hull, i, (0, 255, 0), 1, 8)
    return (bAnd,bAnd_blur,fark)

def yumusatma(img):
    erode = cv2.erode(img, np.ones((1, 3)))
    closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    dilation = cv2.dilate(opening, np.ones((3, 3), np.uint8), iterations=1)
    blurImg = cv2.medianBlur(dilation, 5)
    return(blurImg)

def farkBulma(imgFoto, imgVideo, imgSon):

    bitImgFoto = mask_alma(imgFoto)
    bitImgVideo = mask_alma(imgVideo)
    cv2.imshow("Mask Video",bitImgVideo)
    xorBit = cv2.bitwise_xor(bitImgFoto, bitImgVideo)
    xorBit = yumusatma(xorBit)
    cv2.imshow("Eklenme - Cikma", xorBit)

    pemImgFoto = mask_alma_pembe(imgFoto)
    pemImgVideo = mask_alma_pembe(imgVideo)
    xorPem = cv2.bitwise_xor(pemImgFoto, pemImgVideo)
    xorPem = yumusatma(xorPem)
    beyImgFoto = mask_alma_beyaz(imgFoto)
    beyImgVideo = mask_alma_beyaz(imgVideo)
    xorBey = cv2.bitwise_xor(beyImgFoto, beyImgVideo)
    xorBey = yumusatma(xorBey)

    contours, hierarchy = cv2.findContours(xorBit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    EklenmeCikma = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 500:
            (x, y, w, h) = cv2.boundingRect(contours[i])
            EklenmeCikma.append((x, y, w, h))

    for i in range(len(EklenmeCikma)):
        (x, y, w, h) = EklenmeCikma[i]
        roi = bitImgVideo[y:y + h, x:x + w]
        TP = w * h
        whitePoints = cv2.countNonZero(roi)
        percentage = int(whitePoints / TP * 100)
        #print(percentage)
        if percentage > 10:
            #EKLEME
            cv2.rectangle(imgSon, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
        else:
            #ÇIKARTMA
            cv2.rectangle(imgSon, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 255), 2)

    andRenkDeg = cv2.bitwise_and(xorPem, xorBey)
    andRenkDeg = yumusatma(andRenkDeg)
    cv2.imshow("Renk Degisimi", andRenkDeg)

    contours, hierarchy = cv2.findContours(andRenkDeg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    RenkDegisimi = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 500:
            (x, y, w, h) = cv2.boundingRect(contours[i])
            RenkDegisimi.append((x, y, w, h))
            cv2.rectangle(imgSon, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 0), 2)

    for i in range(len(RenkDegisimi)):
        (x, y, w, h) = RenkDegisimi[i]
        roi = pemImgVideo[y:y + h, x:x + w]
        TP = w * h
        whitePoints = cv2.countNonZero(roi)
        percentage = int(whitePoints / TP * 100)
        #print(percentage)
        if percentage > 10:
            #PEMBE DEĞİŞİMİ
            cv2.rectangle(imgSon, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 0), 2)
        else:
            #BEYAZ DEĞİŞİMİ
            cv2.rectangle(imgSon, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 2)

    return()

#####################################################################

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
        out = cv2.dilate(out,np.ones((3,3),np.uint8),iterations= 3)
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
        out = cv2.dilate(out, np.ones((3, 3), np.uint8), iterations=3)
        output_image = output_image + out

    if show == 1:
        show_image = np.reshape(np.repeat(input_image, 3, axis=1),
                                (input_image.shape[0], input_image.shape[1], 3)) * 255
        show_image[:, :, 1] = show_image[:, :, 1] - output_image * 255
        show_image[:, :, 2] = show_image[:, :, 2] - output_image * 255
        plt.imshow(show_image)

    return output_image  # , np.where(output_image == 1)

#####################################################################

def videoGoster():
    cv2.line(frame, (200, 0), (200, 540), (0, 0, 0), 2)
    cv2.line(frame, (760, 0), (760, 540), (0, 0, 0), 2)
    cv2.putText(frame, "Ekleme", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Cikarma", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Pembeden", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Beyaza", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Beyazdan", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Pembeye", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Video {0}x{1}" .format(frame.shape[1], frame.shape[0]), frame)
    return

def skel(img):

    #erode = cv2.erode(img,np.ones((3,3),np.uint8))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    dilation = cv2.dilate(opening, np.ones((3, 3), np.uint8), iterations=1)
    #erode = cv2.erode(dilation,kernel)
    #cv2.imshow("Maskelenmis 1", dilation)

    binary = dilation > filters.threshold_otsu(dilation)

    skeleton_lee = skeletonize(binary, method='lee')

    lint_img = find_line_intersection(skeleton_lee, 0)

    img1 = np.uint8(lint_img)

    contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    points = []

    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        points.append(((int(x + (w) / 2), int(y + (h) / 2))))
    return (skeleton_lee,lint_img,points)

def siralama(points):

    ErrorCode = 0

    satir1 = []
    satir2 = []

    k = len(points)

    for i in range(k):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        e = 30
        if abs(y2 - y1) < e:
            satir1.append(points[i])
        else:
            satir1.append(points[i])
            sonI = i
            break
    if len(satir1) < 3 or len(satir1) > 3:
        #print("Satır 1 sıkıntılı !! ")
        ErrorCode = 1
        return(points, ErrorCode, satir1, satir2)

    satir2.append(points[sonI + 1])
    for a in range(sonI + 1, k - 1):
        x1, y1 = points[a]
        x2, y2 = points[a + 1]
        e = 40
        if abs(y2 - y1) < e:
            satir2.append(points[a + 1])
        else:
            break

    if len(satir2) < 4:
        #print("Satır 2 sıkıntılı !! ")
        ErrorCode = 2
        return(points, ErrorCode, satir1, satir2)

    satir1.sort()
    satir2.sort()

    #print("satir1 : ", satir1)
    #print("satir2 : ", satir2)

    e = 12

    x1, y1 = satir1[0]
    for a in range(len(satir2)):
        x2, y2 = satir2[a]
        if x1 - e < x2 < x1 + e:
            nokta4 = (x2, y2)
            break
    x1, y1 = satir1[1]
    for a in range(len(satir2)):
        x2, y2 = satir2[a]
        if x1 - e < x2 < x1 + e:
            nokta5 = (x2, y2)
            break
    x1, y1 = satir1[2]
    for a in range(len(satir2)):
        x2, y2 = satir2[a]
        if x1 - e < x2 < x1 + e:
            nokta6 = (x2, y2)
            break
    nokta7list = []
    x1, y1 = nokta6
    for a in range(len(satir2)):
        x2, y2 = satir2[a]
        if y1 - e < y2 < y1 + e and x1 < x2:
            nokta7list.append((x2, y2))
    nokta7list.sort()
    try:
        nokta7 = nokta7list[0]
    except Exception:
        ErrorCode = 7
        return (points, ErrorCode, satir1, satir2)

    pointsSon = []

    for i in range(len(satir1)):
        pointsSon.insert(i, satir1[i])

    pointsSon.insert(3, nokta4)
    pointsSon.insert(4, nokta5)
    pointsSon.insert(5, nokta6)
    pointsSon.insert(6, nokta7)

    return(pointsSon,ErrorCode, satir1, satir2)

iskelet1, kesisim1, points1 = skel(mask_alma(img1))

#cv2.imshow("iskelet",iskelet1)

pointsSon, ErrorCode, satir1, satir2 = siralama(points1)

if ErrorCode == 0:
    loop = True
elif ErrorCode == 1:
    print("İlk görselde satır 1 Sıkıntılı !!")
    cv2.imshow("iskelet", iskelet1)
    loop = False
elif ErrorCode == 2:
    print("İlk görselde satır 2 Sıkıntılı !!")
    cv2.imshow("iskelet", iskelet1)
    loop = False
elif ErrorCode == 7:
    print("İlk görselde nokta 7 Sıkıntılı !!")
    cv2.imshow("iskelet", iskelet1)
    loop = False
else:
    loop = False

#print("points1 : ", points1)
#print("pointsSon : ", pointsSon)

for i in range(len(points1)):
    (x, y) = points1[i]
    cv2.circle(img1, (x, y), 3, (255, 0, 0), 1, -1)
for i in range(len(pointsSon)):
    (x, y) = pointsSon[i]
    image = cv2.putText(img1, str(i + 1), (x + 2, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
for i in range(len(satir1)):
    (x, y) = satir1[i]
    e = 10
    cv2.line(img1, (x + e, y), (x + e, 0), (0, 0, 255), 1)
    cv2.line(img1, (x - e, y), (x - e, 0), (0, 0, 255), 1)

points1 = pointsSon.copy()
cv2.imshow("img1", img1)

#####################################################################

while loop:

    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if ret == 0:
        break
    else:
        pass
        #print("########################################################")

    frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))

    framePembe = mask_alma_pembe(frame)
    contours, hierarchy = cv2.findContours(framePembe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area = []

    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    try:
        bigArea = max(area)
    except:
        cv2.putText(frame, "Alan hesaplanamadı !", (10, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        videoGoster()
        cv2.waitKey(20)
        continue

    bigAreaIndex = area.index(bigArea)
    #print("En buyuk alan: " + str(bigArea))
    if bigArea < 10000:
        cv2.putText(frame, "Alan kucuk !", (10, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        #print("Alan küçük")
        videoGoster()
        cv2.waitKey(20)
        continue

    #print("Alan yeterince büyük.")

    x1, y1, w1, h1 = cv2.boundingRect(contours[bigAreaIndex])

    top_mask = mask_alma(frame)

    if x1 - 10 < 0:
        solDeger = 0
    else:
        solDeger = x1 - 10

    if x1 + w1 + 10 > frame.shape[1]:
        sagDeger = frame.shape[1]
    else:
        sagDeger = x1 + w1 + 10

    if solDeger > 200 and sagDeger < 760:
        pass
    else:
        #print("Alan Dışı !!")
        cv2.putText(frame, "Alan disinda !", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        videoGoster()
        cv2.waitKey(20)
        continue

    roi = top_mask[0:y1 + h1 + 10, solDeger:sagDeger]
    roiF = frame[0:y1 + h1 + 10, solDeger:sagDeger]

    #cv2.imshow("ROI", roiF)

    try:
        iskelet2, kesisim2, points2 = skel(roi)
    except Exception:
        #print("Skeletonize Error")
        cv2.putText(frame, "Skeletonize Error !", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        videoGoster()
        cv2.waitKey(20)
        continue

    #cv2.imshow("Skeletonize", iskelet2)

    if len(points2) < 6 and len(points2) > 15:
        #print("Kesişim noktaları sıkıntılı !! ")
        cv2.putText(frame, "Nokta bulma Error !", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        videoGoster()
        cv2.waitKey(20)
        continue

    pointsSon, ErrorCode, satir1, satir2 = siralama(points2)

    if ErrorCode == 0:
        pass
    elif ErrorCode == 1:
        #print("Videoda satır 1 Sıkıntılı !!")
        cv2.putText(frame, "Satir 1 Error !", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        videoGoster()
        cv2.waitKey(20)
        continue
    elif ErrorCode == 2:
        #print("Videoda satır 2 Sıkıntılı !!")
        cv2.putText(frame, "Satir 2 Error !", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        videoGoster()
        cv2.waitKey(20)
        continue
    elif ErrorCode == 7:
        #print("Videoda nokta 7 Sıkıntılı !!")
        cv2.putText(frame, "Nokta 7 Error !", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        videoGoster()
        cv2.waitKey(20)
        continue

    #print("pointsSon : ", pointsSon)

    points2 = []
    points2 = pointsSon.copy()



    endofF = find_endoflines(iskelet2)

    endofF = np.uint8(endofF)

    contours, hierarchy = cv2.findContours(endofF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    pointsEndofF = []

    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        pointsEndofF.append(((int(x + (w) / 2), int(y + (h) / 2))))



    keypoints1 = np.float32(points1)
    keypoints2 = np.float32(points2)

    try:
        h, mask = cv2.findHomography(keypoints1, keypoints2)
        height, width, channels = roiF.shape
        warped = cv2.warpPerspective(copyimg1, h, (width, height))
    except Exception:
        print("Homografi Bulunamadı !!")
    else:
        added = cv2.addWeighted(bitwise_alma(warped), 0.5, roiF, 0.5, 0)
        cv2.imshow("Added", added)

    try:
        iskeletW, kesisimW, pointsW = skel(mask_alma(warped))
    except Exception:
        print("Skeletonize Warped Error")
        continue

    #cv2.imshow("Skeletonize Warped", iskeletW)

    if len(pointsW) < 6 and len(pointsW) > 15:
        print("Kesişim noktaları warped sıkıntılı !! ")
        videoGoster()
        continue

    pointsW, ErrorCodeW, satir1W, satir2W = siralama(pointsW)

    endofW = find_endoflines(iskeletW)

    endofW = np.uint8(endofW)

    contours, hierarchy = cv2.findContours(endofW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    pointsEndofW = []

    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        pointsEndofW.append(((int(x + (w) / 2), int(y + (h) / 2))))
    """
    for i in range(len(pointsEndofW)):
        (x, y) = pointsEndofW[i]
        cv2.circle(frame, (x+x1-10, y), 7, (0, 255, 0), 1, -1)
    for i in range(len(pointsEndofF)):
        (x, y) = pointsEndofF[i]
        cv2.circle(frame, (x+x1-10, y), 7, (255, 0, 255), 1, -1)
    """
    for i in range(len(pointsEndofW)):
        (xW, yW) = pointsEndofW[i]
        for i in range(len(pointsEndofF)):
            (xF, yF) = pointsEndofF[i]
            if xW - 8 < xF < xW + 8 and yW - 6 < yF < yW +6:
                points2.append((xF, yF))
                pointsW.append((xW, yW))
                break

    keypoints1 = np.float32(pointsW)
    keypoints2 = np.float32(points2)

    try:
        h, mask = cv2.findHomography(keypoints1, keypoints2)
        height, width, channels = roiF.shape
        warped2 = cv2.warpPerspective(warped, h, (width, height))
    except Exception:
        print("Homografi 2 Bulunamadı !!")
    else:

        added2 = cv2.addWeighted(bitwise_alma(warped2), 0.5, roiF, 0.5, 0)
        cv2.imshow("Added 2", added2)
        farkBulma(warped2,roiF,roiF)


    cv2.putText(frame, "OK !", (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    videoGoster()

#####################################################################

cap.release()
cv2.destroyAllWindows()