"""
Met deze tool kan een worp van de bal opgenomen worden.
De UV en Z waarden worden frame per frame opgenomen samen met de corresponderende tijd.
De opname start bij het detecteren van een bal en stopt na een vooraf gedefinieerd aantal succesvolle frames.
"""

#bibliotheken
import numpy
from src import Robot_Ball_Catcher_Functions as func
import os
import time
import cv2 as opencv

#functie om een momentopname op te slaan in de opnamematrix
def save(start,i,pixel,depth):
    #tijd = huidige tijd -starttijd
    t=round(time.time()*1000)-start
    matrix[i,0]=t
    matrix[i,1]=pixel[0]
    matrix[i,2]=pixel[1]
    matrix[i,3]=depth
    #volgende momentopname 1 plaats verder in de array
    i=i+1
    return i

#grootte van de opname
frames=15

matrix=numpy.zeros((frames,4))

#gegevens van hsv kalibratie importeren uit de datafolder
datapath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'
hsvfile=numpy.load(datapath+'\hsv.npy')
# onderste en bovenste kleur definieren
lower_color = numpy.array([hsvfile[0],hsvfile[2],hsvfile[4]])
upper_color = numpy.array([hsvfile[1],hsvfile[3],hsvfile[5]])

#intrinsieke en extrinsieke gegevens importeren
mtx=numpy.load(datapath+'\intrinsics.npy')
dist=numpy.load(datapath+'\distortion.npy')
ext=numpy.load(datapath+'\extrinsics.npy')
rvecs=numpy.load(datapath+'\extrinsic_rvecs.npy')
tvecs=numpy.load(datapath+'\extrinsic_tvecs.npy')

#venster instellingen
window_name='UVZ recorder'
window_b, window_h = 1920, 1080

#camera instellingen
color_resolution = (1920,1080)
depth_resolution = (1280,720)
frames_per_second = 30

#minimum straal van de bal contour
minradius=30

#stel venster en camera in
func.setwindow(window_name,window_b,window_h)
pipeline, align_object = func.startcamera(color_resolution,depth_resolution,frames_per_second)

#momentopname teller start op 0
i=0
#loop
while True:
    #beelden ophalen
    color_image, depth_image = func.getframes(pipeline, align_object)
    #masker van de bal opvragen
    mask=func.getmask(color_image,lower_color,upper_color)
    #pixel van het zwaartepunt van de bal zoekekn
    ball_pixel, radius = func.getballpixel(mask, minradius)
    #wanneer er een bal gevonden is is er een zwaartepunt (pixel), bepaal hier de diepte
    if ball_pixel:
        depth_pixel = func.getdepthpixel(depth_image, ball_pixel)
    else:
        depth_pixel = None
    #visualiseer de gedetecteerde bal en de UV en Z waarde op het kleurenbeeld
    img = func.showballpixel(color_image, ball_pixel, depth_pixel, None, None, None, radius)
    #assen tekenenen
    img = func.drawaxes(img, mtx, dist, rvecs, tvecs, 60)
    #wanneer de bal goed gedetecteerd is, start opname, anders blijf wachten
    if depth_pixel:
        #starttijd
        starttime=round(time.time()*1000)
        while True:
            # beelden ophalen
            color_image, depth_image = func.getframes(pipeline, align_object)
            # masker van de bal opvragen
            mask = func.getmask(color_image, lower_color, upper_color)
            # pixel van het zwaartepunt van de bal zoekekn
            ball_pixel, radius = func.getballpixel(mask, minradius)
            # wanneer er een bal gevonden is is er een zwaartepunt (pixel), bepaal hier de diepte
            if ball_pixel:
                depth_pixel = func.getdepthpixel(depth_image, ball_pixel)
            else:
                depth_pixel = None
            #wanneer de bal goed gedetecteerd is, maak een momentopname
            if depth_pixel:
                i=save(starttime, i, ball_pixel, depth_pixel)
                #wanneer het maximum aantal frames bereikt is, stop met opnemen
                if i==frames:
                    break
                # visualiseer de gedetecteerde bal en de UV en Z waarde op het kleurenbeeld
                img = func.showballpixel(color_image, ball_pixel, depth_pixel, None, None, None, radius)
                # assen tekenenen
                img = func.drawaxes(img, mtx, dist, rvecs, tvecs, 60)
                #projecteer 'RECORDING'
                opencv.putText(color_image, "RECORDING", (10, 500), opencv.FONT_HERSHEY_SIMPLEX,\
                               2, (255, 255, 255), 2, opencv.LINE_AA)
                # beelden laten zien
                func.showimage(img, window_name)
        break

    # beelden laten zien
    func.showimage(img, window_name)
print(matrix)
numpy.save(datapath+'\prerecording.npy',matrix)
print('done')


