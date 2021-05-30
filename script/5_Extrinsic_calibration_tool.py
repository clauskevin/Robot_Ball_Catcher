'''
Met dit script kan de extrinsieke matrix van de camera bepaald worden.
Met de save slider kan deze matrix worden opgeslagen, zo is deze beschikbaar voor andere scripts.
'''
#bibliotheken
import numpy
import cv2 as opencv
import os
from src import Robot_Ball_Catcher_Functions as func

#eigenschappen patroon
h=14
b=9
size=17.6 #aantal mm zijde van een vierkant op het schaakbord

#laden van intrinsieke gegevens uit data
datapath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'
mtx=numpy.load(datapath+'\intrinsics.npy')
dist=numpy.load(datapath+'\distortion.npy')

#venster instellingen
window_name='Extrinsic calibration'
window_b, window_h = 1920, 1080

#camera instellingen
color_resolution = (1920,1080)
depth_resolution = (1280,720)
frames_per_second = 30

#stel venster en camera in
func.setwindow(window_name,window_b,window_h)
pipeline, align_object = func.startcamera(color_resolution,depth_resolution,frames_per_second)

#functie die niets doet, nodig voor sliders
def nothing(*args):
    pass

#aanmaken van slider
opencv.createTrackbar('save',window_name,0,1,nothing)

# termination criteria
criteria = (opencv.TERM_CRITERIA_EPS + opencv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# objectpunten van het schaakbord voorbereiden
objp = numpy.zeros((b*h,3), numpy.float32)
objp[:,:2] = numpy.mgrid[0:b,0:h].T.reshape(-1,2)
objp = 17.6*objp

while True:
    #slider waarde opvragen
    save = opencv.getTrackbarPos('save', window_name)
    #wanneer slider geactiveerd, verlaat loop
    if(save):
        break

    # beelden ophalen
    img, depth_image = func.getframes(pipeline, align_object)

    # grijsschaal beeld maken zodat schaakbordpatroon beter herkend wordt
    gray = opencv.cvtColor(img, opencv.COLOR_BGR2GRAY)
    # zoek hoekpunten op schaakbord (+ de 3 bollen moeten aanwezig zijn voor consistente orientatie)
    ret, corners = opencv.findChessboardCornersSB(gray, (b, h), opencv.CALIB_CB_MARKER)
    #als de hoekpunten gevonden zijn
    if ret == True:
        #verfijn de hoekpunten tot subpixels
        corners2 = opencv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #bepaal de rotatie en translatievectoren naar het patroon
        ret, rvecs, tvecs, inliers = opencv.solvePnPRansac(objp, corners2, mtx, dist)
        #visualiseer de hoekpunten op het schaakpatroon
        opencv.drawChessboardCorners(img,(b,h),corners2, ret)
        #teken de assen op het schaakpatroon
        img=func.drawaxes(img,mtx,dist,rvecs,tvecs,60)
        #projecteer de waarden voor de translatie en rotatie vectoren
        rvecs_as_string = str(numpy.round(rvecs[0],2))+'   '+str(numpy.round(rvecs[1],2))+\
                          '   '+str(numpy.round(rvecs[2],2))
        tvecs_as_string = str(numpy.round(tvecs[0],2))+'   '+str(numpy.round(tvecs[1],2))+\
                          '   '+str(numpy.round(tvecs[2],2))
        opencv.putText(img, "rvecs:   " + rvecs_as_string, (10, 300), opencv.FONT_HERSHEY_SIMPLEX,\
                       1, (255, 255, 255), 2, opencv.LINE_AA)
        opencv.putText(img, "tvecs:   " + tvecs_as_string, (10, 400), opencv.FONT_HERSHEY_SIMPLEX,\
                       1, (255, 255, 255), 2, opencv.LINE_AA)

    #laat de beelden zien
    func.showimage(img, window_name)

#converteer rotatievector naar rotatiematrix en bouw de extrinsieke matrix
rvecs_matrix=opencv.Rodrigues(rvecs)[0]
extrinsics = numpy.hstack((rvecs_matrix, tvecs))
extrinsics = numpy.vstack((extrinsics, [0.0, 0.0, 0.0, 1.0]))
#print de extrinsieke matrix
print(extrinsics)
#sla de extrinsieke gegevens op in numpy arrays
numpy.save(datapath+'\extrinsics.npy',extrinsics)
numpy.save(datapath+'\extrinsic_rvecs.npy',rvecs)
numpy.save(datapath+'\extrinsic_tvecs.npy',tvecs)
print('done')