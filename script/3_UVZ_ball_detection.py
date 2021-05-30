"""
Met deze tool wordt de baldetectie gevisualiseerd
en wordt de pixelcoördinaat UV en diepte Z geprojecteerd op het kleurbeeld.
"""

#bibliotheken
import numpy
from src import Robot_Ball_Catcher_Functions as func
import os


#gegevens van hsv kalibratie importeren uit de datafolder
datapath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'
hsvfile=numpy.load(datapath+'\hsv.npy')
# onderste en bovenste kleur definieren
lower_color = numpy.array([hsvfile[0],hsvfile[2],hsvfile[4]])
upper_color = numpy.array([hsvfile[1],hsvfile[3],hsvfile[5]])

#venster instellingen
window_name='UVZ ball detection'
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
    #beelden laten zien
    func.showimage(img, window_name)