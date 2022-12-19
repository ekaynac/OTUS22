import cv2

# For performance videopath and savepath better be in same drive
videopath = "F:/SabitKanatVideolar/Labellanacakvideolar/GH010012_Trim.mp4"
savepath = "C:/Users/enes_/Desktop/teest"
frameRate = 0.5 #//it will capture image in each 0.5 second

vidcap = cv2.VideoCapture(videopath)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(savepath+"/frame%d.jpg" % count, image)     # save frame as JPG file
    return hasFrames
sec = 0
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)