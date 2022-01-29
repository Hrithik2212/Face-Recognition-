import cv2 as cv 
#We are defining a function to return the overlarged picture to a veiwable size
def frame_rescale(frame,scale=0.4):
    #this function works for images,videos aand live cam 
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)
    return cv.resize(frame , dimensions , interpolation=cv.INTER_AREA )


 

