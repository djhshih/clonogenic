from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from functools import reduce


# Show each color channel
def show_color_channels(cimg):
    cv.imshow("blue", cimg[:,:,0])
    cv.imshow("green", cimg[:,:,1])
    ctv.imshow("red", cimg[:,:,2])

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
    tsimg = cv.morphologyEx(tsimg, cv.MORPH_DILATE, kernel)
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
    #cv.imshow("certain_bg", certain_bg)
    #cv.waitKey(0)

    # get bona fide foreground based on distance transform
    dist = cv.distanceTransform(timg, cv.DIST_L2, 0)
    dist = np.uint8(dist / dist.max() * 255)
    #cv.imshow("dist", dist)
    #cv.waitKey(0)
    _, certain_fg = cv.threshold(dist, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    certain_fg = np.uint8(certain_fg)
    #cv.imshow("certain_fg", certain_fg)
    #cv.waitKey(0)

    # get unknown, border regions
    unknown = cv.subtract(certain_bg, certain_fg)
    #cv.imshow("unknown", unknown)
    #cv.waitKey(0)

    # label the pixels (baackground will be labeled as 0)
    _, markers = cv.connectedComponents(certain_fg)

    # ensure that background is labeled 1
    markers = markers + 1

    # mark unknown regions as 0
    markers[unknown > 0] = 0
    #cv.imshow("markers", cv.applyColorMap(np.uint8(markers), cv.COLORMAP_JET))
    #cv.waitKey(0)

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

def find_circles_hough(wimg, wcimg):
    circles = cv.HoughCircles(wimg, cv.HOUGH_GRADIENT, 1, 3,
        param1 = 30, param2 = 20,
        minRadius = 3, maxRadius = 20)
    if circles is not None:
        circles = circles[0]
        circles = np.uint16(np.around(circles))
        markers = np.int8(np.ones(wimg.shape))
        # draw circles
        for c in circles:
            x0, y0, r = c
            cv.circle(wcimg, (x0, y0), r, (0, 255, 0), 2)
            cv.circle(markers, (x0, y0), r, -1, 1)
    return (circles, markers)

def apply_circular_mask(wimg, r, shrink=0):
    mask = np.zeros(wimg.shape, np.uint8)
    cv.circle(mask, (r, r), r-shrink, 255, -1)
    return cv.bitwise_and(wimg, wimg, mask=mask)

def apply_wells_imask(img, wells, shrink=0):
    mask = np.ones(img.shape[0:2], np.uint8)
    for w in wells:
        x0, y0, r = w
        cv.circle(mask, (x0, y0), r-shrink, 0, -1)
    return cv.bitwise_and(img, img, mask=mask)

def apply_wells_mask(img, wells, shrink=0):
    mask = np.zeros(img.shape[0:2], np.uint8)
    for w in wells:
        x0, y0, r = w
        cv.circle(mask, (x0, y0), r-shrink, 255, -1)
    return cv.bitwise_and(img, img, mask=mask)

# find local maxima
def find_local_maxima(wimg, threshold=5):
    # using blurred image worsen results here
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    peak = cv.dilate(wimg, kernel, iterations=2)
    peak = cv.subtract(peak, wimg)
    _, peak = cv.threshold(peak, threshold, 255, cv.THRESH_BINARY)
    cv.imshow("peak", peak)
    cv.waitKey(0)

    flat = cv.erode(wimg, kernel, iterations=2)
    flat = cv.subtract(wimg, flat)
    _, flat = cv.threshold(flat, threshold, 255, cv.THRESH_BINARY)
    flat = cv.bitwise_not(flat)
    cv.imshow("flat", flat)
    cv.waitKey(0)

    peak[flat > 0] = 255
    local = cv.bitwise_not(peak)
    return local


##############################################################################

import argparse

pr = argparse.ArgumentParser("Count colonies for clonogenic assay")
pr.add_argument("input", help="input image file")

argv = pr.parse_args()

cimg = cv.imread(argv.input)
cv.imshow("original", cimg)

cimg_raw = cimg.copy()

print(cimg.shape)

# convert to grayscale, invert, and denoise
img = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
img = cv.bitwise_not(img)
img = cv.medianBlur(img, 5)
cv.imshow("gray", img)
cv.waitKey(0)

wells = find_wells(img, (2,3))
print(wells)

margin = 8

img = apply_wells_mask(img, wells, margin)
cimg = apply_wells_mask(cimg, wells, margin)
cv.imshow("bg_masked", img)
cv.waitKey(0)

mode_hue = find_mode_hue(cimg)

#tsimg = filter_bgr_similarity(cimg, (95, 0, 27))
#cv.imshow("bgr_sim_threshold", tsimg)
#cv.waitKey(0)



well = wells[0]



wimg = select_well(img, well)
wcimg = select_well(cimg, well)

cv.imshow("well", wimg)
cv.waitKey(0)

tsimg = filter_hsv_similarity(wcimg, mode_hue)
cv.imshow("hsv_sim_threshold", tsimg)
cv.waitKey(0)

radius = well[2]
margin2 = 24

marked = wcimg.copy()

markers1 = mark_boundaries(tsimg, marked)
marked[markers1 == -1] = (255, 255, 0)
cv.imshow("marked1", marked)
cv.waitKey(0)


local = find_local_maxima(wimg)
local = apply_circular_mask(local, well[2], shrink=margin2)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
local = cv.morphologyEx(local, cv.MORPH_CLOSE, kernel)
cv.imshow("local", local)
cv.waitKey(0)

markers2 = mark_boundaries(local, wcimg)
marked[markers2 == -1] = (255, 0, 255)
cv.imshow("marked2", marked)
cv.waitKey(0)

margin3 = 12

timg = cv.adaptiveThreshold(wimg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -2)
timg = apply_circular_mask(timg, radius, shrink=margin2)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
timg = cv.morphologyEx(timg, cv.MORPH_OPEN, kernel)
cv.imshow("athreshold", timg)
cv.waitKey(0)

markers3 = mark_boundaries(timg, wcimg)
marked[markers3 == -1] = (0, 255, 255)
cv.imshow("marked3", marked)
cv.waitKey(0)


circles, markers4 = find_circles_hough(wimg, marked)
cv.imshow("marked4", marked)
cv.waitKey(0)

markers_set = [markers1, markers2, markers3, markers4]

# get combined borders
#borders = reduce(cv.bitwise_or, [np.uint8(x == -1) * 255 for x in markers_set])
#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
#borders = cv.dilate(borders, kernel)
#cv.imshow("borders", borders)
#cv.waitKey(0)


def fill_in_markers(markers, value=255):
    tmarkers = np.uint8(markers > 0) * 255
    tmarkers = cv.bitwise_not(tmarkers)
    # fill in the contours
    _, contours, _ = cv.findContours(tmarkers, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #_, contours, _ = cv.findContours(tmarkers, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # fill in the contour
    tmarkers = cv.drawContours(tmarkers, contours, -1, value, thickness=-1)
    # remove the border
    tmarkers = cv.drawContours(tmarkers, contours, -1, 0, thickness=1, lineType=cv.LINE_8)
    return tmarkers

# get combined marker contour
#kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
tmarkers = reduce(cv.add, [fill_in_markers(x) for x in markers_set])
cv.imshow("tmarkers", tmarkers)
cv.waitKey(0)

#tmarkers = reduce(cv.add, [cv.erode(fill_in_markers(x, 1), kernel) for x in markers_set])
#ttmarkers = cv.adaptiveThreshold(tmarkers, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
#        cv.THRESH_BINARY, 7, 1)
#ttmarkers = find_local_maxima(tmarkers, threshold=0)
#cv.imshow("ttmarkers", ttmarkers)
#cv.waitKey(0)


def refine(tmarkers, border):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    border = cv.dilate(border, kernel)
    #border = cv.GaussianBlur(border, (9,9), 0)
    cv.imshow("border!", border)
    cv.waitKey(0)

    #timg = tmarkers
    timg = cv.subtract(tmarkers, np.uint8(border))
    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    #timg = cv.morphologyEx(timg, cv.MORPH_CLOSE, kernel)
    cv.imshow("timg!", timg)
    cv.waitKey(0)

    # get bona fide background
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    certain_bg = cv.dilate(timg, kernel, iterations=3)
    cv.imshow("certain_bg!", certain_bg)
    cv.waitKey(0)

    #dist = cv.distanceTransform(timg, cv.DIST_L2, 0)
    #dist = np.uint8(dist / dist.max() * 255)
    #cv.imshow("dist!", dist)
    #cv.waitKey(0)
    #_, certain_fg = cv.threshold(dist, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #certain_fg = np.uint8(certain_fg)
    #cv.imshow("certain_fg!", certain_fg)
    #cv.waitKey(0)
    certain_fg = timg

    # get unknown, border regions
    unknown = cv.subtract(certain_bg, certain_fg)
    #unknown = cv.bitwise_or(unknown, border)
    cv.imshow("unknown!", unknown)
    cv.waitKey(0)

    _, markers = cv.connectedComponents(certain_fg, 4)
    markers = markers + 1
    markers[unknown > 0] = 0
    cv.imshow("markers!", cv.applyColorMap(np.uint8(markers), cv.COLORMAP_JET))
    cv.waitKey(0)

    markers = cv.watershed(wcimg, markers)

    marked = wcimg.copy()
    marked[markers == -1] = (255, 255, 0)
    cv.imshow("marked!", marked)
    cv.waitKey(0)

    tmarkers = fill_in_markers(markers)
    cv.imshow("tmarkers!", tmarkers)
    cv.waitKey(0)
    return tmarkers

#border = np.uint8(markers1 == -1) * 255;
#tmarkers = refine(tmarkers, border)

_, contours, _ = cv.findContours(tmarkers, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

marked = wcimg.copy()
marked = cv.drawContours(marked, contours, -1, (255, 255, 0), thickness=1)
cv.imshow("marked", marked)
cv.waitKey(0)


cc1 = cv.connectedComponentsWithStats(fill_in_markers(markers1), 4, cv.CV_32S)
cc4 = cv.connectedComponentsWithStats(fill_in_markers(markers4), 4, cv.CV_32S)

# get areas of high confidence colonies
areas = np.concatenate((cc1[2][1:, cv.CC_STAT_AREA], cc4[2][1:, cv.CC_STAT_AREA]))
area_mean = np.mean(areas)
area_sd = np.std(areas)
print("mean", area_mean, ", sd:", area_sd)

cc = cv.connectedComponentsWithStats(tmarkers, 4, cv.CV_32S)

min_size = 9

# centroids[0] is the background
stats = cc[2][1:]
centroids = cc[3][1:]

idx = stats[:, cv.CC_STAT_AREA] > min_size
print(stats[:, cv.CC_STAT_AREA])

ncolonies = 0
for i in range(len(centroids)):
    c = centroids[i]
    area = stats[i, cv.CC_STAT_AREA]
    x0, y0 = np.uint16(np.around(c))
    if area > min_size:
        # TODO ideally, we would use intensity instead...
        if area < area_mean + 2*area_sd:
            n = 1
        else:
            # truncate
            n = np.uint8(np.around(area / area_mean))
        ncolonies += n
        cv.circle(marked, (x0, y0), n, (0, 255, 255), -1)
     
cv.imshow("final", marked)
cv.waitKey(0)

print("number of colonies:", ncolonies)

fg_max_area = np.sum(apply_circular_mask(np.ones(wimg.shape), radius, shrink=margin))

# subtract background
mask = tmarkers
bg = cv.bitwise_and(wimg, wimg, mask=cv.bitwise_not(tmarkers))
bg_mean = np.mean(bg)
fg = cv.subtract(cv.bitwise_and(wimg, wimg, mask=tmarkers), bg_mean)
cv.imshow("fg", fg)
cv.waitKey(0)

area = np.sum(fg > 0) / fg_max_area

print("area: ", area)

s = apply_circular_mask(hsv_color_similarity(wcimg, mode_hue), radius, shrink=margin)
outside = apply_wells_imask(cimg_raw, wells, shrink=-margin)

os = hsv_color_similarity(outside, mode_hue)
cv.imshow("os", os)
cv.waitKey(0)

sbg = np.mean(os)
sbg_sd = np.std(os)
print("sbg: ", sbg)

fg = s > sbg + 6 * sbg_sd
cv.imshow("fg2", fg / fg.max() * 255)
cv.waitKey(0)

area2 = np.sum(fg) / fg_max_area
print("area2: ", area2)

# background has been subtracted from both background,
# so it is sufficient to just sum the foreground
# max value of each pixel is 255
intensity = np.float(np.sum(fg)) / 255 / fg_max_area
print("intensity: ", intensity)

intensity2 = np.float(np.sum(cv.subtract(s, sbg))) / fg_max_area
print("intensity2: ", intensity2)


cv.destroyAllWindows()

