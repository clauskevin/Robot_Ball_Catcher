'''
Dit script laat het kleurbeeld en een kleurenkaart van het dieptebeeld zien.
'''

#bibliotheken
import cv2 as opencv
import numpy
from src import Robot_Ball_Catcher_Functions as func

#venster instellingen
window_name='Camera viewer'
window_b, window_h = 960, 1080

#camera instellingen
color_resolution = (1920,1080)
depth_resolution = (1280,720)
frames_per_second = 30

#stel venster en camera in
func.setwindow(window_name,window_b,window_h)
pipeline, align_object = func.startcamera(color_resolution,depth_resolution,frames_per_second)

#loop
while True:
    #beelden ophalen
    color_image, depth_image = func.getframes(pipeline, align_object)
    #kleurenkaart toepassen op het dieptebeeld
    depth_colormap = opencv.applyColorMap(opencv.convertScaleAbs(depth_image, alpha=0.03), opencv.COLORMAP_HSV)
    #kleur- en dieptebeeld boven elkaar zetten
    images = numpy.vstack((color_image, depth_colormap))
    #beelden laten zien
    func.showimage(images, window_name)