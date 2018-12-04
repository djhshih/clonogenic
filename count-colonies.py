import numpy as np
import cv2 as cv

cimg = cv.imread("EPSON005.TIF")

cv.imshow("image", cimg)
#cv.imshow("blue", cimg[:,:,0])
#cv.imshow("green", cimg[:,:,1])
#cv.imshow("red", cimg[:,:,2])
cv.waitKey(0)

# convert to grayscale
img = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
img = cv.medianBlur(img, 5)

hsv = cv.cvtColor(cimg, cv.COLOR_BGR2HSV)
#cv.imshow("hsvb", hsv)

lower = np.array((120, 50, 50))
upper = np.array((170, 255, 255))
mask_color = cv.inRange(hsv, lower, upper)
kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
mask_color = cv.morphologyEx(mask_color, cv.MORPH_OPEN, kernel)
vimg = cv.bitwise_not(img)
simg = cv.bitwise_and(vimg, vimg, mask=mask_color)
cv.imshow("selected_colour", simg)
cv.waitKey(0)

ref_color = (95, 0, 27)
max_dist = np.sqrt(3.0 * 255**2)

cimgf = cimg.astype(np.float)
delta_b = cimgf[:,:,0] - ref_color[0]
delta_r = cimgf[:,:,1] - ref_color[1]
delta_g = cimgf[:,:,2] - ref_color[2]
delta = np.sqrt(np.multiply(delta_b, delta_b) + np.multiply(delta_r, delta_r) + np.multiply(delta_g, delta_g))
sim = 1 - delta / max_dist

sim_img = np.uint8(sim * 255)
cv.imshow("colour_sim", sim_img)
cv.waitKey(0)


tsim = cv.adaptiveThreshold(sim_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -2)
cv.imshow("colour_sim_threshold", tsim)
cv.waitKey(0)


#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
#seimg = cv.morphologyEx(simg, cv.MORPH_ERODE, kernel)
#cv.imshow("bona_fide", seimg)
#cv.waitKey(0)


print(img.shape)

height= img.shape[0]
width = img.shape[1]

plate_nrows = 2
plate_ncols = 3

max_radius = int(min((width / plate_ncols), (height / plate_nrows)) / 2)
print(max_radius)

wells = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 2, max_radius,
    param1 = 100, param2 = 100,
    minRadius = int(max_radius / 2), maxRadius = max_radius)[0]
wells = np.uint16(np.around(wells))
print(wells)

# TODO sort wells by coordinates

for w in wells:
    x0, y0, r = w
    # draw the perimeter
    cv.circle(cimg, (x0, y0), r, (0, 255, 0), 2)
    # draw the center
    cv.circle(cimg, (x0, y0), 2, (0, 0, 255), 3)

cv.imshow("Detected wells", cimg)
cv.waitKey(0)


def process_well_hough(img, well):
    x0, y0, r = well
    wimg = img[(y0-r):(y0+r), (x0-r):(x0+r)]
    wcimg = cimg[(y0-r):(y0+r), (x0-r):(x0+r)]

    circles = cv.HoughCircles(wimg, cv.HOUGH_GRADIENT, 1, 3,
        param1 = 30, param2 = 20,
        minRadius = 3, maxRadius = 20)
    print(circles)

    if circles is not None:
        circles = circles[0]
        circles = np.uint16(np.around(circles))
        print(circles)
        for c in circles:
            x0, y0, r = c
            cv.circle(wcimg, (x0, y0), r, (255, 0, 0), 2)

    cv.imshow("well", wcimg)
    cv.waitKey(0)

#process_well_hough(wells[0])


well = wells[1]
x0, y0, r = well

wimg = img[(y0-r):(y0+r), (x0-r):(x0+r)]
wimg = cv.bitwise_not(wimg)

wcimg = cimg[(y0-r):(y0+r), (x0-r):(x0+r)]

def apply_circular_mask(wimg, border=10):
    mask = np.zeros(wimg.shape, np.uint8)
    cv.circle(mask, (r, r), r-border, 255, -1)
    return cv.bitwise_and(wimg, mask)

cv.imshow("well", wimg)
cv.waitKey(0)

#wimg = apply_circular_mask(wimg)

bimg = cv.GaussianBlur(wimg, (7, 7), 0)
cv.imshow("blur", bimg)
cv.waitKey(0)

fimg = np.fft.fftshift(cv.dft(np.float32(wimg), flags = cv.DFT_COMPLEX_OUTPUT))
lmag = np.log(1 + cv.magnitude(fimg[:,:,0], fimg[:,:, 1]))
#from matplotlib import pyplot as plt
#plt.imshow(lmag, cmap="gray")
#plt.show()
#cv.waitKey(0)

hp_sigma = 3
lp_sigma = 60

wheight = wimg.shape[0]
wwidth = wimg.shape[1]
ksize = min(wheight, wwidth)
hpf = cv.getGaussianKernel(ksize, hp_sigma)
hpf = hpf / hpf.max()
hpf = 1 - (hpf * hpf.T)
print("hpf: ", hpf)
cv.imshow("highpass_filter", hpf)
cv.waitKey(0)

lpf = cv.getGaussianKernel(ksize, lp_sigma)
lpf = lpf * lpf.T
lpf = lpf / lpf.max()
cv.imshow("lowpass_filter", lpf)
cv.waitKey(0)

gpf = np.multiply(lpf, hpf)
print("min: {}, max: {}".format(gpf.min(), gpf.max()))
cv.imshow("filter", gpf / gpf.max())
cv.waitKey(0)

fimgf = np.dstack((np.multiply(fimg[:,:,0], gpf), fimg[:,:,1]))

gimg = cv.idft(np.fft.ifftshift(fimgf))
#gimg = cv.idft(np.fft.ifftshift(fimg))
print(gimg.dtype)
print(gimg.shape)
gimg = cv.magnitude(gimg[:,:,0], gimg[:,:,1])
gimg = np.uint8((gimg / gimg.max()) * 255)
print(gimg.max())
cv.imshow("gaussian_bandpass_filter", gimg)
cv.waitKey(0)




lg = cv.Laplacian(gimg, cv.CV_16S, 11)
#kernel = np.ones((3,3))
kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
# use erode to find min of neighbourhood
minlg = cv.morphologyEx(lg, cv.MORPH_ERODE, kernel)
# use dilate to find max of neighbourhood
maxlg = cv.morphologyEx(lg, cv.MORPH_DILATE, kernel)
# zero crossing occurs at a positive pixel with an adjacent negative pixel or
# a negative pixel with an adjacent positive pixel 
print(np.histogram(lg))
threshold = 3
zcross = np.logical_or(
    np.logical_and(minlg < threshold, lg > threshold),
    np.logical_and(maxlg > threshold, lg < threshold))

cv.imshow("zero_crossing", zcross.astype(np.uint8)*255)
cv.waitKey(0)

fimg = wimg.astype(np.float)

m = cv.blur(fimg, (3, 3))
m2 = cv.blur(np.multiply(fimg, fimg), (3, 3))
sd = cv.sqrt(m2 - m*m)

cv.imshow("sd", sd)
cv.waitKey(0)

# edge occurs at Laplacian zero-crossings in regions of high local variance
ledge = np.logical_and(sd > 2, zcross).astype(np.uint8)*255
cv.imshow("laplacian_edge", ledge)
cv.waitKey(0)


#kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
ledge = cv.morphologyEx(ledge, cv.MORPH_CLOSE, kernel)
#ledge = cv.medianBlur(ledge, 5)
cv.imshow("laplacian_edge", ledge)
cv.waitKey(0)


ledge2, contours0, hierarchy = cv.findContours(ledge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(hierarchy)

cv.imshow("contours", ledge2)
cv.waitKey(0)

contours = [cv.approxPolyDP(c, 3, True) for c in contours0]
print(contours)
vis = cv.drawContours(np.zeros((wheight, wwidth), np.uint8), contours, -1, 255, thickness=-1)
cv.imshow("filled_contours", vis)
cv.waitKey(0)



cedge = cv.Canny(bimg, 30, 15, 3)
cv.imshow("canny_edge", cedge)
cv.waitKey(0)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
cedge = cv.morphologyEx(cedge, cv.MORPH_CLOSE, kernel)
cv.imshow("canny_edge", cedge)
cv.waitKey(0)


cedge2, contours0, hierarchy = cv.findContours(cedge, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

cv.imshow("contours", cedge2)
cv.waitKey(0)

contours = [cv.approxPolyDP(c, 1, True) for c in contours0]
print("hierarchy", hierarchy)
vis = cv.drawContours(np.zeros((wheight, wwidth), np.uint8), contours, -1, 255, thickness=-1)
cv.imshow("canny_filled_contours", vis)
cv.waitKey(0)



#lg_abs = cv.convertScaleAbs(lg, alpha = 255/(2*np.max(lg)))
#lg_abs = apply_circular_mask(lg_abs)
#lg = cv.threshold(lg_abs, 10, 255, cv.THRESH_BINARY)[1]

#print(np.mean(lg_abs))
print(zcross)

#cv.imshow("lglacian", lg == 0)
#cv.imshow("lglacian", lg_abs)

#dimg = wimg - bimg
#cv.imshow("diff1", dimg)
#cv.waitKey(0)

#dimg = bimg - wimg
#cv.imshow("diff", dimg)
#cv.waitKey(0)

timg = cv.adaptiveThreshold(wimg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -2)
cv.imshow("athreshold", timg)
cv.waitKey(0)

timg = cv.medianBlur(timg, 5)
#timg = cv.morphologyEx(timg, cv.MORPH_OPEN, np.ones((3,3)))
cv.imshow("athreshold", timg)
cv.waitKey(0)

cc = cv.connectedComponentsWithStats(timg, 8, cv.CV_32S)

centroids = cc[3]

for c in centroids:
    x0, y0 = np.uint16(np.around(c))
    cv.circle(wcimg, (x0, y0), 2, (0, 0, 255), 2)

cv.imshow("athreshold", wcimg)
cv.waitKey(0)


cv.destroyAllWindows()
