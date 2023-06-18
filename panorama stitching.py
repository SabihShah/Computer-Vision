import cv2
import matplotlib.pyplot as plt
image_paths=['img 1.png', 'img 2.png', 'img 3.png', 'img 4.png', 'img 5.png']
# initialized a list of images
imgs = []

for i in range(len(image_paths)):
	imgs.append(cv2.imread(image_paths[i]))
	
print(len(imgs))


stitchy=cv2.Stitcher.create()
(dummy,output)=stitchy.stitch(imgs)

if dummy != cv2.STITCHER_OK:
	print("stitching ain't successful")
else:
	print('Your Panorama is ready!!!')

# final output
cv2.imshow('final result',output)

cv2.waitKey(0)
