import cv2


class Preprocess:
	(sizex, sizey) = (320, 160)

	@staticmethod
	def preprocess(image):
		return image
		
	    #leny  = image.shape[0]
	    #image = image[leny * .35 : leny *.85 ,:]
	    #image = cv2.resize(image, (Preprocess.sizex, Preprocess.sizey))
	    #return image
