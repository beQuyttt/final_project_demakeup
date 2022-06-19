import cv2, glob
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('video/out.mp4', fourcc, 40, (390, 350))
for i in range (284):
   path = 'output/' + str(i) + '.png'
   img = cv2.resize(cv2.imread(path),(390,350))
   video.write(img)
video.release()
