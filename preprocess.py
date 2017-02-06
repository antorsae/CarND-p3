import cv2


class Preprocess:
	(sizex, sizey) = (200, 66)

	@staticmethod
	def preprocess(image):
	    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
	    leny  = image.shape[0]
	    image = image[leny * .35 : leny *.85 ,:]
	    image = cv2.resize(image, (Preprocess.sizex, Preprocess.sizey))
	    return image
