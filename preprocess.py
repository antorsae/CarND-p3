import cv2

sizex = 200 #320
sizey = 66 #160

def size():
	return (sizex, sizey)

def preprocess(image):
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    leny  = image.shape[0]
    image = image[leny * .35 : leny *.85 ,:]
    image = cv2.resize(image, (sizex, sizey))
    return image
