from fmtm import fmtm
import cv2
import time
img_org = "OralB_toothbrush\OralB_B8.jpg"
tm1 = fmtm()
# tm1.new_template(img_org, (300, 400), 'type1',(-60,60), 1) # size = (w, h), shape = (h, w, c)
# tm1.make_template((122,36,30,312)) # (y,x,h,w)
# tm1.make_roi((1,2,26,57))
# tm1.make_mask((1,2,26,57),(1,150,26,57))
# tm1.rotate()
# tm1.save_config()
start_time = time.time()
tm1.load_template('type1')
tm1.fmatch('OralB_B11.jpg',(-60,60),0.06)
# tm1.rerotate_crop()
print("--- %s seconds ---" % (time.time() - start_time))
