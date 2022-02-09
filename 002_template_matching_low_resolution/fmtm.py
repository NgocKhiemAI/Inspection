import os
import cv2
import numpy as np
import imageio
from scipy import misc, ndimage
import time
import utils

class fmtm:
    # Init Function
    def __init__(self):
        self.DIR = "template"                   # alarm
        self.TEMPLATE_DIR = "template"
        self.MASK_DIR = "mask"                  # alarm
        self.IMG_DIR = "data"
        self.RESULT_DIR = "result"
        # defind step to find object
        self.big_step = 20  
        self.med_step = 4
        self.sma_step = 1

        pass

    def new_template(self, img_org, size, NAME, rot_range, res):
        self.img_org = cv2.resize(cv2.imread(img_org, 0), size)
        self.size = self.img_org.shape # (h,w)
        self.NAME = NAME
        if not os.path.isdir(os.path.join(self.DIR, self.NAME)):
            os.mkdir(os.path.join(self.DIR, self.NAME))
        if not os.path.isdir(os.path.join(self.DIR, self.NAME, self.TEMPLATE_DIR)):
            os.mkdir(os.path.join(self.DIR, self.NAME, self.TEMPLATE_DIR))
        if not os.path.isdir(os.path.join(self.DIR, self.NAME, self.MASK_DIR)):
            os.mkdir(os.path.join(self.DIR, self.NAME, self.MASK_DIR))
        self.rot_range = rot_range
        self.res = res
        self.is_new_template_flg = True

    def make_template(self, loc):
        x, y, w, h = loc
        self.base_template = self.img_org[y:y+h,x:x+w]
        cv2.imwrite(os.path.join(self.DIR, self.NAME, self.TEMPLATE_DIR, "base_template.png"),self.base_template)

    def make_roi(self, loc):
        x, y, w, h = loc
        self.roi_loc = loc
        self.roi = self.base_template[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(self.DIR, self.NAME, self.TEMPLATE_DIR, "roi.png"),self.roi)
        

    def make_mask(self, *args):     # *arg để gọi vào bao nhiêu tham số cũng được
        self.base_mask = np.zeros(self.base_template.shape, dtype=np.uint8)
        self.origin_mask = np.ones(self.base_template.shape, dtype=np.uint8)*255
        if len(args) == 0:
            self.base_mask[:,:] = 255
        else:
            for loc in args:
                x, y, w, h = loc
                self.base_mask[y:y+h, x:x+w] = 255
        cv2.imwrite(os.path.join(self.DIR, self.NAME, self.MASK_DIR, "base_mask.png"), self.base_mask)
        cv2.imwrite(os.path.join(self.DIR, self.NAME, self.MASK_DIR, "origin_mask.png"), self.origin_mask)

    def rotate(self):
        try:
            base_template = imageio.imread(os.path.join(self.DIR, self.NAME, self.TEMPLATE_DIR, 'base_template.png'))
            base_mask = imageio.imread(os.path.join(self.DIR, self.NAME, self.MASK_DIR, 'base_mask.png'))
            origin_mask = imageio.imread(os.path.join(self.DIR, self.NAME, self.MASK_DIR, 'origin_mask.png'))
        except IOError:
            print('Failed to rotate templates. Base template is not found.')
            return
        for deg in range(self.rot_range[0], self.rot_range[1]+1, self.res):
            template = ndimage.rotate(base_template, deg)
            mask = ndimage.rotate(base_mask, deg)
            omask = ndimage.rotate(origin_mask, deg)
            index = str(deg) if deg >= 0 else 'n' + str(abs(deg))
            imageio.imsave(os.path.join(self.DIR, self.NAME, self.TEMPLATE_DIR,'template_' + index + '.png'), template)
            imageio.imsave(os.path.join(self.DIR, self.NAME, self.MASK_DIR, 'mask_' + index + '.png'), mask)
            imageio.imsave(os.path.join(self.DIR, self.NAME, self.MASK_DIR, 'omask_' + index + '.png'), omask)

    def save_config(self):
        with open(os.path.join(self.DIR, self.NAME,self.NAME + '.config'), 'w') as f:
            f.write(','.join(str(i) for i in self.roi_loc))
            f.write('\n')
            f.write(','.join(str(i) for i in self.size))
            f.write('\n')
            f.write(','.join(str(i) for i in self.rot_range))
            f.write('\n')
            f.write(str(self.res))
            f.write('\n')

    def load_template(self, name):
        self.NAME = name
        with open(os.path.join(self.DIR, self.NAME,self.NAME + '.config'), 'r') as f:
            lines = f.readlines()
            self.roi_loc = tuple(map(int, lines[0].split(',')))
            self.size = tuple(map(int, lines[1].split(',')))
            self.rot_range = tuple(map(int, lines[2].split(',')))
            self.res = int(lines[3])

    def fmatch(self, img_name,rotRange,thres):
        # Resize the image to trained size
        self.img_process = cv2.resize(cv2.imread(os.path.join(self.IMG_DIR, img_name),0),(self.size[1],self.size[0]))
        # Apply template matching with big step to get smaller range
        medRange = self.__matchStep(rotRange, self.big_step)
        # Apply template matching with med step to get smaller range
        smaRange = self.__matchStep(medRange, self.med_step)
        # Apply template matching with sma step and threshold to get bounding boxes
        boxesWithDeg = self.__match(smaRange,self.sma_step, thres)
        # Apply non max suppression and get deg
        box, pick = utils.nms(boxesWithDeg[:,:4], 0.5)
        self.resultBox = box.flatten()
        self.resultDeg = int(boxesWithDeg[pick, 4])
        # Draw rectangle
        cv2.rectangle(self.img_process, (int(self.resultBox[0]), int(self.resultBox[1])), (int(self.resultBox[2]), int(self.resultBox[3])), 0, 1)
        # Save the result
        img_name = "".join(img_name.split('.')[:-1])
        with open(os.path.join(self.RESULT_DIR, img_name+'.txt'), 'w') as f:
            f.write(','.join(str(i) for i in box))
            f.write('\n')
            f.write(str(self.resultDeg))
            f.write('\n')
        cv2.imwrite(os.path.join(self.RESULT_DIR, img_name+'.jpg'), self.img_process)
        self.resultName = img_name+'.jpg'
        self.save_result()
        return


    def save_result(self):
        index = self.__getIndexWithDeg(self.resultDeg)
        result = imageio.imread(os.path.join(self.RESULT_DIR, self.resultName))
        print(self.resultBox)
        result = result[self.resultBox[1]:self.resultBox[3],self.resultBox[0]:self.resultBox[2]]
        theomask = imageio.imread(os.path.join(self.DIR, self.NAME, self.MASK_DIR, 'omask_'+index+'.png'))
        rotated = ndimage.rotate(result, -self.resultDeg)
        theomask_rotated = ndimage.rotate(theomask, -self.resultDeg)
        rect = cv2.boundingRect(theomask_rotated) 
        imageio.imsave(os.path.join(self.RESULT_DIR, 'rotated.jpg'), rotated)
        imageio.imsave(os.path.join(self.RESULT_DIR, 'rotatedMask.jpg'), theomask_rotated)
        cropped_img = rotated[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
        imageio.imsave(os.path.join(self.RESULT_DIR, 'cropped.jpg'), cropped_img)
        x,y,w,h = self.roi_loc
        roi = cropped_img[y:y+h, x:x+w]
        imageio.imsave(os.path.join(self.RESULT_DIR, 'croppedRoi.jpg'), roi)

    # Private methods
    def __getTemplateAndMaskWithIndex(self,index):
        # Read the template and mask with specific deg
        tmpl = cv2.imread(os.path.join(self.DIR, self.NAME, self.TEMPLATE_DIR,'template_' + index + '.png'),0)
        mask = cv2.imread(os.path.join(self.DIR, self.NAME, self.MASK_DIR, 'omask_' + index + '.png'), 0)
        return tmpl, mask

    def __getIndexWithDeg(self,deg):
        # Use for negative deg -> n
        return str(deg) if deg >= 0 else 'n' + str(abs(deg))

    def __matchStep(self,rotRange, step):
        scoreMat = np.array([])
        for deg in range(rotRange[0], rotRange[1] + 1, step):
            # For each deg, read the template and mask
            index = self.__getIndexWithDeg(deg)
            tmpl, mask = self.__getTemplateAndMaskWithIndex(index)
            # Apply template matching for the img, get score matrix
            res = cv2.matchTemplate(self.img_process, tmpl, cv2.TM_SQDIFF_NORMED, mask=mask)
            # Calculate average of N minimum values in the matrix
            ave = utils.getAverageNMinimumValues(res, 10)
            scoreMat = utils.stackV(scoreMat,np.array([deg, ave]))
        # Get the deg that has the smallest average score 
        min = scoreMat[utils.getNMinimumIndices(scoreMat[:,1], 1).flatten(), 0]
        # Return smaller rotation range
        lower = int(min[0] - step/2)
        upper = int(min[0] + step/2)
        return (lower, upper)
    
    def __match(self, rotRange, step, thres):
        boxes = np.array([]) # [xStart, yStart, xEnd, yEnd,]
        for deg in range(rotRange[0], rotRange[1] + 1, step):
            # For each deg, read the template and mask
            index = self.__getIndexWithDeg(deg)
            tmpl, mask = self.__getTemplateAndMaskWithIndex(index)
            # Apply template matching for the img
            res = cv2.matchTemplate(self.img_process, tmpl, cv2.TM_SQDIFF_NORMED, mask=mask)
            # Get the bounding boxes' location if the result score smaller than threshold
            loc = np.where(res < thres)
            w, h = tmpl.shape[::-1]
            for pt in zip(*loc[::-1]):
                boxes = utils.stackV(boxes,np.array([pt[0], pt[1], pt[0] + w, pt[1] + h, deg])) # (xStart, yStart, xEnd, yEnd)
        return boxes



