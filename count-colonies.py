from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv


# Show each color channel
def show_color_channels(cimg):
    cv.imshow("blue", cimg[:,:,0])
    cv.imshow("green", cimg[:,:,1])
    cv.imshow("red", cimg[:,:,2])

# identify most common hue from among saturated pixels
def find_mode_hue(cimg):
    hsv = cv.cvtColor(cimg, cv.COLOR_BGR2HSV)
    # filter for pixels with > 50% saturation
    idx = hsv[:,:,1] > 255/2
    cv.imshow("saturated", cv.bitwise_and(cimg, cimg, mask=np.uint8(idx)))
    cv.waitKey(0)
    hsv_hist = np.histogram(hsv[:,:,0][idx], bins=np.arange(180+1))
    return hsv_hist[1][np.argmax(hsv_hist[0])]

# Filter image for a color range
# @param cimg   BGR color image
# @param hue    hue value in range [0, 180]
#                (note: the common range is [0, 360])
# @param tol    tolerance level in (0, 1)
# @param lower  lower bound in HSV space
# @param upper  upper bound in HSV space
def filter_hsv_range(cimg, hue=None, lower=None, upper=None, tol=0.1):
    # use HSV color space to select desired colours
    hsv = cv.cvtColor(cimg, cv.COLOR_BGR2HSV)
    img = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
    if hue is None:
        hue = 90
    delta = 180 * tol
    if lower is None:
        lower = (max(0, hue - delta), 50, 50)
    if upper is None:
        upper = (min(180, hue + delta), 255, 255)
    mask_color = cv.inRange(hsv, lower, upper)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    mask_color = cv.morphologyEx(mask_color, cv.MORPH_OPEN, kernel)
    vimg = cv.bitwise_not(img)
    return cv.bitwise_and(vimg, vimg, mask=mask_color)

def bgr_color_euclidean_similarity(cimg, ref_color):
    cimgf = cimg.astype(np.float)
    delta_b = (cimgf[:,:,0] - ref_color[0]) / 255.0
    delta_r = (cimgf[:,:,1] - ref_color[1]) / 255.0
    delta_g = (cimgf[:,:,2] - ref_color[2]) / 255.0
    delta = np.sqrt(
        np.multiply(delta_b, delta_b) + 
        np.multiply(delta_r, delta_r) + 
        np.multiply(delta_g, delta_g))
    return 1 - delta / np.sqrt(3.0)

def bgr_color_maxabs_similarity(cimg, ref_color):
    cimgf = cimg.astype(np.float)
    delta_b = (cimgf[:,:,0] - ref_color[0]) / 255.0
    delta_r = (cimgf[:,:,1] - ref_color[1]) / 255.0
    delta_g = (cimgf[:,:,2] - ref_color[2]) / 255.0
    delta = np.maximum(
        np.maximum(
            np.abs(delta_b),
            np.abs(delta_r)),
        np.abs(delta_g))
    return 1 - delta

def hsv_color_similarity(cimg, ref_hue):
    hsv = cv.cvtColor(cimg, cv.COLOR_BGR2HSV).astype(np.float)
    # similarity in hue
    hue_sim = 1 - np.abs(hsv[:,:,0] - ref_hue) / 180.0
    # scale by saturation
    return hue_sim * (hsv[:,:,1]/255.0)

def filter_bgr_similarity(cimg, ref_color):
    s = bgr_color_euclidean_similarity(cimg, ref_color)
    simg = np.uint8(s * 255)
    _, tsimg = cv.threshold(simg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    tsimg = cv.morphologyEx(tsimg, cv.MORPH_OPEN, kernel)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    tsimg = cv.morphologyEx(tsimg, cv.MORPH_CLOSE, kernel)
    return tsimg

def subtract_edges(img):
#    bsimg = cv.blur(img, (9, 9), 0)
#    cv.imshow("bsimg", bsimg)
#    cv.waitKey(0)
#    dimg = cv.subtract(bsimg, simg)
#    cv.imshow("dimg", dimg)
#    cv.waitKey(0)
#    print(dimg)
#    _, tdimg = cv.threshold(dimg, 20, 255, cv.THRESH_BINARY_INV);
#    cv.imshow("tdimg", tdimg)
#    cv.waitKey(0)
    timg = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
        cv.THRESH_BINARY_INV, 11, 2)
    timg = cv.dilate(timg, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    return cv.subtract(img, timg)

def filter_hsv_similarity(cimg, ref_hue):
    s = hsv_color_similarity(cimg, ref_hue)
    simg = np.uint8(s * 255)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    simg = cv.morphologyEx(simg, cv.MORPH_OPEN, kernel)
    cv.imshow("hsv_sim", simg)
    cv.waitKey(0)
    # subtract edges
    dimg = subtract_edges(simg)
    cv.imshow("dimg", dimg)
    cv.waitKey(0)
    # threshold
    _, tsimg = cv.threshold(dimg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    tsimg = cv.morphologyEx(tsimg, cv.MORPH_OPEN, kernel)
    return tsimg

# Keep high-confidence hue foreground
def keep_hue_foreground(cimg, ref_hue):
    # filter for target hue
    simg = filter_hsv_range(cimg, ref_hue)
    cv.imshow("hue_filtered", simg)
    cv.waitKey(0)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    seimg = cv.morphologyEx(simg, cv.MORPH_ERODE, kernel)
    return seimg

# Split colonies
# @param timg  thresholded image
# @param cimg  original color image
# @return  markers matrix where boundary regions are marked as -1
def mark_boundaries(timg, cimg):
    # get bona fide background
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    certain_bg = cv.dilate(timg, kernel, iterations=3)
    cv.imshow("certain_bg", certain_bg)
    cv.waitKey(0)

    # get bona fide foreground based on distance transform
    dist = cv.distanceTransform(timg, cv.DIST_L2, 0)
    dist = np.uint8(dist / dist.max() * 255)
    cv.imshow("dist", dist)
    cv.waitKey(0)
    _, certain_fg = cv.threshold(dist, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    certain_fg = np.uint8(certain_fg)
    cv.imshow("certain_fg", certain_fg)
    cv.waitKey(0)

    # get unknown, border regions
    unknown = cv.subtract(certain_bg, certain_fg)
    cv.imshow("unknown", unknown)
    cv.waitKey(0)

    # label the pixels (baackground will be labeled as 0)
    _, markers = cv.connectedComponents(certain_fg)

    # ensure that background is labeled 1
    markers = markers + 1

    # mark unknown regions as 0
    markers[unknown > 0] = 0
    cv.imshow("markers", cv.applyColorMap(np.uint8(markers), cv.COLORMAP_JET))
    cv.waitKey(0)

    # mark boundary regions with -1
    markers = cv.watershed(cimg, markers)
    return markers

# TODO Fix result for non-square image!
#      resulting appears to have a rotated image superimposed
def gaussian_bandpass_filter(img, hp_sigma=3, lp_sigma=60):
    fimg = np.fft.fftshift(cv.dft(np.float32(img), flags = cv.DFT_COMPLEX_OUTPUT))
    print(fimg.shape)
    #lmag = np.log(1 + cv.magnitude(fimg[:,:,0], fimg[:,:, 1]))
    #from matplotlib import pyplot as plt
    #plt.imshow(lmag, cmap="gray")
    #plt.show()
    #cv.waitKey(0)

    ksize = min(img.shape[0], img.shape[1]) 
    hpf = cv.getGaussianKernel(ksize, hp_sigma)
    hpf = hpf / hpf.max()
    hpf = 1 - (hpf * hpf.T)
    #cv.imshow("highpass_filter", hpf)
    #cv.waitKey(0)

    lpf = cv.getGaussianKernel(ksize, lp_sigma)
    lpf = lpf * lpf.T
    lpf = lpf / lpf.max()
    #cv.imshow("lowpass_filter", lpf)
    #cv.waitKey(0)

    gpf = np.multiply(lpf, hpf)
    # NB funny things happen when img is *not* square!
    gpf = cv.resize(gpf, (img.shape[1], img.shape[0]))
    #print("min: {}, max: {}".format(gpf.min(), gpf.max()))
    #cv.imshow("filter", gpf / gpf.max())
    #cv.waitKey(0)

    # apply bandpass filter to the real component,
    # leaving the imaginary component alone
    fimgf = np.dstack((np.multiply(fimg[:,:,0], gpf), fimg[:,:,1]))

    # apply inverse Fourier transform to recover the filtered image
    gimg = cv.idft(np.fft.ifftshift(fimgf))
    gimg = cv.magnitude(gimg[:,:,0], gimg[:,:,1])
    gimg = np.uint8((gimg / gimg.max()) * 255)
    return gimg

def local_sd(img, ksize=(3,3)):
    fimg = img.astype(np.float)
    # mean
    m = cv.blur(fimg, ksize)
    # mean squared
    m2 = cv.blur(np.multiply(fimg, fimg), ksize)
    # NB numerically instable
    sd = cv.sqrt(m2 - m*m)
    return sd

def laplacian_edge(img, zcross_threshold=3, sd_ksize=(3,3), sd_threshold=2):
    # find zero crossings after Laplacian transform
    lg = cv.Laplacian(gimg, cv.CV_16S, 11)
    #kernel = np.ones((3,3))
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    # use erode to find min of neighbourhood
    minlg = cv.morphologyEx(lg, cv.MORPH_ERODE, kernel)
    # use dilate to find max of neighbourhood
    maxlg = cv.morphologyEx(lg, cv.MORPH_DILATE, kernel)
    # zero crossing occurs at a positive pixel with an adjacent negative pixel or
    # a negative pixel with an adjacent positive pixel 
    zcross = np.logical_or(
        np.logical_and(minlg < zcross_threshold, lg > zcross_threshold),
        np.logical_and(maxlg > zcross_threshold, lg < zcross_threshold))
    #cv.imshow("zero_crossing", zcross.astype(np.uint8)*255)
    #cv.waitKey(0)

    sd = local_sd(img, sd_ksize)
    #cv.imshow("sd", sd)
    #cv.waitKey(0)

    # edge occurs at Laplacian zero-crossings in regions of high local variance
    ledge = np.logical_and(sd > sd_threshold, zcross).astype(np.uint8)*255
    return ledge

# Sort wells by y and x coordinates, allowing for imprecision in coordinates
# @param wells  list of list [x0, y0, r]
def sort_wells(wells):
    xs = np.sort([w[0] for w in wells])
    ys = np.sort([w[1] for w in wells])

    # find mean radius
    r = np.mean([w[2] for w in wells])

    # Discover grid coordinates
    # starting with the smallest coordinate,
    # serially find the next coordinate that is sufficiently far (2r)
    # from the previous coordinate
    def find_coords(ps):
        coords = [ps[0]]
        cur_p = ps[0]
        if len(ps) > 1:
            for p in ps[1:]:
                if p > cur_p + 2*r:
                    coords.append(p)
                    cur_p = p
        return coords

    coords_x = find_coords(xs)
    coords_y = find_coords(ys)

    # Snap a position to the first sufficiently near coordinate
    def snap(p, coords):
        for c in coords:
            if abs(p - c) < r:
                return c

    # sort wells by the snapped coordinates 
    return sorted( wells, key = lambda well:
        (snap(well[1], coords_y), snap(well[0], coords_x)) )

# @return list of wells;
#         each well is of the form (x0, y0, r),
#         where (x0, y0) is the center and r is the radius
def find_wells(img, plate_shape):
    height = img.shape[0]
    width = img.shape[1]
    max_radius = int(min(
        (img.shape[1] / plate_shape[1]),
        (img.shape[0] / plate_shape[0])) / 2)
    wells = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 2, max_radius,
        param1 = 100, param2 = 100,
        minRadius = int(max_radius / 2), maxRadius = max_radius)[0]
    wells = np.uint16(np.around(wells))
    return sort_wells(wells)

def show_wells(cimg, wells):
    for w in wells:
        x0, y0, r = w
        # draw the perimeter
        cv.circle(cimg, (x0, y0), r, (0, 255, 0), 2)
    cv.imshow("Detected wells", cimg)
    cv.waitKey(0)

def select_well(img, well):
    x0, y0, r = well
    return img[(y0-r):(y0+r), (x0-r):(x0+r)]

def process_well_hough(img, cimg, well):
    x0, y0, r = well
    wimg = img[(y0-r):(y0+r), (x0-r):(x0+r)]
    wcimg = cimg[(y0-r):(y0+r), (x0-r):(x0+r)]

    circles = cv.HoughCircles(wimg, cv.HOUGH_GRADIENT, 1, 3,
        param1 = 30, param2 = 20,
        minRadius = 3, maxRadius = 20)
    if circles is not None:
        circles = circles[0]
        circles = np.uint16(np.around(circles))
        # draw circles
        for c in circles:
            x0, y0, r = c
            cv.circle(wcimg, (x0, y0), r, (0, 255, 0), 2)
        #cv.imshow("well", wcimg)
        #cv.waitKey(0)
    return circles

def apply_well_mask(wimg, r, shrink=5):
    mask = np.zeros(wimg.shape, np.uint8)
    cv.circle(mask, (r, r), r-shrink, 255, -1)
    return cv.bitwise_and(wimg, wimg, mask=mask)

def apply_wells_mask(img, wells, shrink=5):
    mask = np.zeros(img.shape[0:2], np.uint8)
    for w in wells:
        x0, y0, r = w
        cv.circle(mask, (x0, y0), r-shrink, 255, -1)
    return cv.bitwise_and(img, img, mask=mask)


cimg = cv.imread("EPSON005.TIF")
cv.imshow("original", cimg)

print(cimg.shape)

# convert to grayscale, invert, and denoise
img = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
img = cv.bitwise_not(img)
img = cv.medianBlur(img, 5)
cv.imshow("gray", img)
cv.waitKey(0)

mode_hue = find_mode_hue(cimg)

wells = find_wells(img, (2,3))
print(wells)

img = apply_wells_mask(img, wells, 8)
cimg = apply_wells_mask(cimg, wells, 8)
cv.imshow("bg_masked", img)
cv.waitKey(0)


#tsimg = filter_bgr_similarity(cimg, (95, 0, 27))
#cv.imshow("bgr_sim_threshold", tsimg)
#cv.waitKey(0)

tsimg = filter_hsv_similarity(cimg, mode_hue)
cv.imshow("hsv_sim_threshold", tsimg)
cv.waitKey(0)

marked = cimg.copy()
markers = mark_boundaries(tsimg, marked)
marked[markers == -1] = (0, 0, 255)
cv.imshow("watershed", marked)
cv.waitKey(0)



well = wells[0]

process_well_hough(img, cimg, well)



wimg = select_well(img, well)
wcimg = select_well(cimg, well)


cv.imshow("well", wimg)
cv.waitKey(0)

#wimg = apply_circular_mask(wimg)

bimg = cv.GaussianBlur(wimg, (7, 7), 0)
cv.imshow("blur", bimg)
cv.waitKey(0)

gimg = gaussian_bandpass_filter(wimg)
cv.imshow("gaussian_bandpass_filter", gimg)
cv.waitKey(0)


ledge = laplacian_edge(wimg)
cv.imshow("laplacian_edge", ledge)
cv.waitKey(0)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
ledge = cv.morphologyEx(ledge, cv.MORPH_CLOSE, kernel)
cv.imshow("laplacian_edge", ledge)
cv.waitKey(0)


ledge2, contours0, hierarchy = cv.findContours(ledge, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#print(hierarchy)

cv.imshow("contours", ledge2)
cv.waitKey(0)

contours = [cv.approxPolyDP(c, 3, True) for c in contours0]
vis = cv.drawContours(np.zeros(wimg.shape, np.uint8), contours, -1, 255, thickness=-1)
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
#print("hierarchy", hierarchy)
vis = cv.drawContours(np.zeros(wimg.shape, np.uint8), contours, -1, 255, thickness=-1)
cv.imshow("canny_filled_contours", vis)
cv.waitKey(0)

timg = cv.adaptiveThreshold(wimg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -2)
cv.imshow("athreshold", timg)
cv.waitKey(0)

#timg = cv.medianBlur(timg, 5)
#timg = cv.morphologyEx(timg, cv.MORPH_ERODE, kernel)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
timg = cv.morphologyEx(timg, cv.MORPH_OPEN, kernel)
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

