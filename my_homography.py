import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.core.numeric import zeros_like
import scipy
from matplotlib import pyplot as plt

import pickle as pk
import os
# #Add imports if needed:
#     """
#     Your code here
#     """
# #end imports

# #Add extra functions here:
#     """
#     Your code here
#     """
# #Extra functions end

# # HW functions:
def getPoints(im1,im2,N):
    
    
    pts1 = []
    pts2 = []
    for i in range(N):
        plt.imshow(im1)
        pts1 += plt.ginput(n = 1, timeout =0)
        plt.close()
    
    
        plt.imshow(im2)
        pts2 += plt.ginput(n = 1, timeout =0)
        plt.close()
    
    p1 = np.zeros((2, N))
    p2 = np.zeros((2, N))
    
    
    for i in range(N):
        p1[:, i] = pts1[i]
        p2[:, i] = pts2[i]
    
    return p1,p2



    
    
def computeH(p1, p2):
    
    if p1.shape[0] != 2:
        p1 = p1.T
    if p2.shape[0] != 2:
        p2 = p2.T
    
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)


    N = p1.shape[1]
    A = np.zeros((2*N , 9), dtype=np.float64)
    
    A[0: 2*N: 2] = np.vstack([p1[0,:], 
                            p1[1,:], 
                            np.ones_like(p1[1,:]) , 
                            np.zeros_like(p1[1,:]),
                            np.zeros_like(p1[1,:]),
                            np.zeros_like(p1[1,:]),
                            -p1[0,:]*p2[0,:], 
                            -p1[1,:]*p2[0,:], 
                            -p2[0, :]]).T
    
    A[1: 2*N: 2] = np.vstack([np.zeros_like(p1[1,:]), 
                          np.zeros_like(p1[1,:]), 
                          np.zeros_like(p1[1,:]), 
                          p1[0,:], 
                          p1[1,:], 
                          np.ones_like(p1[1,:]), 
                          -p1[0,:]*p2[1,:],
                          -p1[1,:]*p2[1,:], 
                          -p2[1, :]]).T
    
    
    (U,D,Vh) = np.linalg.svd(A, False)
    h = Vh.T[:,-1]
    
    H2to1 = np.reshape(h, (3,3))
    
    return H2to1





def warpH(im1, H, out_size):
    #im1 = cv2.copyMakeBorder(im1, 700, 1000, 2000, 1000, cv2.BORDER_CONSTANT, value=0)
    warp_im = cv2.warpPerspective(im1, H, dsize=out_size, flags =  cv2.INTER_CUBIC)
    
    return warp_im

def imageStitching(img1, wrap_img2, pano_dim):
    
    pano_width, pano_height = pano_dim    
    borders = np.nonzero(wrap_img2)
    border_x_min = borders[1].min()

    panoImg = np.zeros((pano_dim[1] , pano_dim[0], 3))
    
    panoImg[:pano_height, border_x_min:, :] = wrap_img2[:pano_height, border_x_min:, :]
    panoImg[:pano_height, :img1.shape[1], :] = img1[:pano_height, :, :]
    
    return panoImg.astype(np.uint8)



# im1 = cv2.imread('data/incline_L.png')

# im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
# im2 = cv2.imread('data/incline_R.png')


# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

# p1, p2 = getPoints(im1, im2, 8)

# H = computeH(p1, p2)
# im2_wrap = warpH(im2, H, (im1.shape[1] + im2.shape[1] , im1.shape[0] + im2.shape[0]))


# plt.imshow(im2_wrap)
# plt.show()

# pano_img = imageStitching(im1, im2_wrap)
# plt.imshow(pano_img)
# plt.show()

def ransacH(matches, locs1, locs2, nIter, tol):
    """
    Your code here
    """
    return bestH

def getPoints_SIFT(im1, im2):
    descriptor = cv2.xfeatures2d.SIFT_create()
    
    
    (kps1, features_im1) = descriptor.detectAndCompute(im1, None)
    (kps2, features_im2) = descriptor.detectAndCompute(im2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    rawMatches = bf.knnMatch(features_im1, features_im2, 2)
    
    

    ratio_thresh = 0.2
    matches = []
    for m,n in rawMatches:
        if m.distance < ratio_thresh * n.distance:
            # p1.append(np.flip(np.array(kps1[m.queryIdx].pt, dtype=np.float32)))
            # p2.append(np.flip(np.array(kps2[m.trainIdx].pt, dtype=np.float32)))
            
            matches.append(m)
        
    kpsA = np.float32([kp.pt for kp in kps1])
    kpsB = np.float32([kp.pt for kp in kps2])

    p1 = np.float32([kpsA[m.queryIdx] for m in matches])
    p2 = np.float32([kpsB[m.trainIdx] for m in matches])
        
    return p1, p2

# load all the sintra and beach imgs
def load_imgs():
    beach_imgs = []
    sintra_imgs = []
    for f in os.listdir(r'data'):
        if 'beach' in f:
            im = cv2.imread(os.path.join('data', f))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            beach_imgs.append(im)
        
        if 'sintra' in f:
            im = cv2.imread(os.path.join('data', f))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            sintra_imgs.append(im)
    
    return sintra_imgs, beach_imgs

def load_points(path):
    f = open(path, 'rb')
    pts = pk.load(f)
    sintra_points = pts['sintra']
    beach_points = pts['beach']
    
    beach_points_buff = []
    sintra_points_buff = []
    
    for (key1, val1), (key2, val2) in zip(beach_points.items(), sintra_points.items()):
        beach_points_buff.append(val1)
        
        sintra_points_buff.append(val2)

    
    return sintra_points_buff, beach_points_buff

def get_boarders(im, H):

    rows, cols = im.shape[:2]
    rect = [[0,0, 1], [cols-1, 0, 1], [0, rows-1, 1], [cols-1,rows-1, 1]]
    
    transformed_rect = []
    for pt in rect:
        trans_point = H@(np.float32(pt).T)
        transformed_rect += [trans_point / trans_point[2]]
        
    # transformed_rect = cv2.transform(rect, H)
    
    sorted_x = sorted(transformed_rect, key=lambda pt: pt[0] )
    sorted_y = sorted(transformed_rect, key=lambda pt: pt[1] )
    
    minx, miny = sorted_x[0][0], sorted_y[0][1]
    
    maxx, maxy = sorted_x[-1][0], sorted_y[-1][1]
    
    
    # return transformd ROI: x,y,width,height
    new_rect = (int(minx), int(miny) ,int(maxx - minx ) , int(maxy - miny))
    return new_rect


# correct the calculated H transformation so we wont loose negative indexes
def correct_H(new_roi , H):
    
    x,y, width, height = new_roi
    
    # cordinates where the new matrix is copied too
    xpos, ypos = 0,0
    if x < 0:
        xpos = -x

    if y < 0:
        ypos = -y
    
    T = np.float32([[1,0, xpos], [0, 1, ypos], [0, 0, 1]])
    
    H_corr = T.dot(H)
    
    return H_corr, (xpos, ypos)


def pad_image(new_roi, im1):
    xmin, ymin, width, height = new_roi
    
    xmax = xmin + width
    ymax = ymin + height
    
    im_height, im_width = im1.shape[:2]
    
    
    pad_left = -xmin if xmin < 0 else 0
    
    pad_right = xmax-im_width if xmax-im_width > 0 else 0
    
    pad_top = -ymin if ymin < 0 else 0
    
    pad_bottom = ymax-im_height if ymax-im_height > 0 else 0
    
    im1 = cv2.copyMakeBorder(im1, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value= 0)
    
    
    return im1

def stitch(im1, im2, pos, im1_shape):
    pos_x, pos_y = pos
    im1_height, im1_width = im1_shape
    
    roi_im1 = im1[pos_y: pos_y+ im1_height ,pos_x: pos_x + im1_width ]
    
    locs1 = np.where(cv2.cvtColor(roi_im1, cv2.COLOR_RGB2GRAY) != 0)
    non_zero_idx = np.nonzero(im1)
    im2[non_zero_idx] = im1[non_zero_idx]

    # plt.imshow(im2)
    # plt.show()
    return im2

# only the non-zero pixels are weighted to the average
def mean_blend(img1, img2):
    assert(img1.shape == img2.shape)
    locs1 = np.where(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) != 0)
    blended1 = np.copy(img2)
    blended1[locs1[0], locs1[1]] = img1[locs1[0], locs1[1]]
    locs2 = np.where(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) != 0)
    blended2 = np.copy(img1)
    blended2[locs2[0], locs2[1]] = img2[locs2[0], locs2[1]]
    blended = cv2.addWeighted(blended1, 1, blended2, 0, 0)
    return blended

def warpPano(prevPano, img, pos):

    xpos, ypos = pos

    idx = np.s_[ypos : ypos + img.shape[0], xpos : xpos + img.shape[1]]
    
    rect = (xpos, ypos, img.shape[1], img.shape[0])

    prevPano = pad_image(rect, prevPano)

    prevPano[idx] = mean_blend(prevPano[idx], img)
    # crop extra paddings
    x, y, w, h = cv2.boundingRect(cv2.cvtColor(prevPano, cv2.COLOR_RGB2GRAY))
    result = prevPano[y : y + h, x : x + w]
    # return the resulting image with shift amount
    return (result, (xpos - x, ypos - y))

def scale_cords(pts: list, scale):
    
    
    pts_scaled = np.array(pts) / scale
    
    return pts_scaled

 
def wrap_multiple_imgs_manually(img_buffer: list, points: list, anchor_im_idx: int=2, mode='horiz'):
    assert len(points) == len(img_buffer) - 1

    final_im = np.copy(img_buffer[2])

    
    
    
    stitch_order=[[3,4],[1,0]]
    pt_batch_indexes = [2, 3, 1, 0]
    anchor_rect = (0,0, final_im.shape[1], final_im[0])
    
    batch_out = []
    
    
    curr_pt = 0
    for batch in stitch_order[:]:
        pos_calc = (0, 0)
        H = None
        final_im = np.copy(img_buffer[2])
        curr_im_idx = anchor_im_idx
        for im_idx in batch:

            pt_batch_idx = pt_batch_indexes[curr_pt] 
            if H is not None:
                pts2 = []

                # apply the image transformation because our refrence points have been transformed
                for pt_idx in range(len(points[pt_batch_idx][0][0])):
                    pt = np.float32([points[pt_batch_idx][0][0][pt_idx], points[pt_batch_idx][0][1][pt_idx], 1])
                    
                    if im_idx < curr_im_idx:
                        trans_point = np.linalg.inv(H)@pt
                        
                    else:
                        trans_point = H@pt
                
                    pts2 += [trans_point[:2] / trans_point[2]]
                    
                pts2 = np.array(pts2).T
            else:
                pts2 = points[pt_batch_idx][0]        
            
            pts1 = points[pt_batch_idx][1]
            #pts2 = points[pt_batch_idx][0]
            
            ###################################################################################
            # Calculate the inverse transform if we stitch 2->1 instead of 1->2 for example...#
            ###################################################################################
            if im_idx < curr_im_idx:
                pts1, pts2 = pts2, pts1
                
            H = computeH(pts1, pts2)
            

            im2 = img_buffer[im_idx]
            
            #final_im = img_buffer[i]
            rect2_borders = get_boarders(im2, H)

            H_curr, pos = correct_H(rect2_borders, H)

                

            wrap_shape = rect2_borders[2], rect2_borders[3]
            
            #im2 = pad_image(rect2_borders, im2)
            im2_wrap = warpH(im2, H_curr, wrap_shape)

            
            plt.imshow(im2_wrap)
            plt.show()
            
            final_im = pad_image(rect2_borders, final_im)
            

            im2_wrap, pos_t = warpPano(final_im, im2_wrap, pos_calc)
            
            pos_calc = (pos[0] + pos_t[0], pos[1] + pos_t[1])
            

            final_im = np.copy(im2_wrap)
            
            # plt.imshow(final_im)
            # plt.show()
            
            curr_pt += 1
        
        batch_out += [final_im]

    
    
    _,_, width, height = anchor_rect
    
    
    # merge_width = batch_out[0].shape[1] + batch_out[1].shape[1]
    # merge_height = batch_out[0].shape[0]
    # merged_stitched = np.zeros((merge_height, merge_width,3))
    
    
    # merged_stitched[:,:batch_out[0].shape[1] - width, :] = batch_out[0][:,:batch_out[0].shape[1] - width, :] 
    
    # merged_stitched[:,batch_out[0].shape[1] - width:, :] = batch_out[1][:,:, :] 
    
    plt.imshow(batch_out[1])
    plt.show()
    return final_im

## merge multiple image ##
sintra_imgs, beach_imgs = load_imgs()

sintra_pts, beach_pts = load_points(r'data/points.pkl') 


# get_boarders(sintra_pts[0])

final_im = wrap_multiple_imgs_manually(sintra_imgs, sintra_pts)
plt.imshow(final_im)
plt.show()

# im1 = cv2.imread(r'data/sintra1.JPG')
# im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

# im2 = cv2.imread(r'data/sintra2.JPG')
# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
# #pts_1, pts_2 = getPoints_SIFT(im1, im2)
# sintra_pts, _ = load_points(r'data/points.pkl')

# # sintra_pts = sintra_pts[0]

# # tt = 1

# # #my_pts1, my_pts2 = getPoints(im1, im2, 10) 
# # H = computeH(sintra_pts[1], sintra_pts[0])


# # rect1_borders = [(0,0), (0, im1.shape[1]), (im1.shape[0], 0), (im1.shape[0], im1.shape[1])]
# # rect2_borders = get_boarders(im2, H)

# # H = correct_H(rect2_borders, H)

# # wrap_shape = int(rect2_borders[2] - rect2_borders[0]), int(rect2_borders[3] - rect2_borders[1])
# # im2_wrap = warpH(im2, H, wrap_shape)

# # im1 = pad_image(rect2_borders, im1)


# plt.subplot(1,2,1)
# plt.imshow(im2_wrap)
# plt.subplot(1,2,2)
# plt.imshow(im1)
# plt.show()

# non_zero_idx = np.nonzero(im1)
# im2_wrap[non_zero_idx] = im1[non_zero_idx]
# plt.imshow(im2_wrap)
# plt.show()



if __name__ == '__main__':
    print('my_homography')
    im1 = cv2.imread('data/incline_L.png')
    im2 = cv2.imread('data/incline_R.png')
    """
    Your code here
    """
