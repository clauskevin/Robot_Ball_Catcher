"""
Met deze tool kan er worden gezocht naar het kleurbereik in de HSV-kleurenruimte van een gekleurd voorwerp.
Aan de hand van 6 sliners kunnen de H, S en V boven- en ondergrenzen aangepast worden.
Ook worden er 3 beelden weergegeven:
Het volledige camerabeeld, het kleurmasker in zwart-wit en het masker uitgewerkt op het camerabeeld.
Verder worden de onderste en bovenste HSV kleur en het gemiddelde HSV kleur weergegeven.
Met de save slider kunnen de resultaten worden opgeslagen voor gebruik in de andere scripts.
"""

#bibliotheken
import cv2 as opencv
import numpy
from src import Robot_Ball_Catcher_Functions as func
import os


#gegevens van vorige kalibratie importeren uit de datafolder
datapath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'
hsvfile=numpy.load(datapath+'\hsv.npy')

#venster instellingen
window_name='HSV calibration tool'
window_b, window_h = 1500, 1080

#camera instellingen
color_resolution = (640,480)
depth_resolution = (640,480)
frames_per_second = 30

#stel venster en camera in
func.setwindow(window_name,window_b,window_h)
pipeline, align_object = func.startcamera(color_resolution,depth_resolution,frames_per_second)

#functie die niets doet, nodig om sliders aan te maken
def nothing (*args):
    pass
# aanmaken van de sliders
opencv.createTrackbar('Hmin', window_name, hsvfile[0], 179, nothing)
opencv.createTrackbar('Hmax', window_name, hsvfile[1], 179, nothing)
opencv.createTrackbar('Smin', window_name, hsvfile[2], 255, nothing)
opencv.createTrackbar('Smax', window_name, hsvfile[3], 255, nothing)
opencv.createTrackbar('Vmin', window_name, hsvfile[4], 255, nothing)
opencv.createTrackbar('Vmax', window_name, hsvfile[5], 255, nothing)
opencv.createTrackbar('save',window_name,0,1,nothing)

# formaat van de beelden definieren
HSVmin = numpy.zeros((color_resolution[1], color_resolution[0], 3), numpy.uint8)
HSVmax = numpy.zeros((color_resolution[1], color_resolution[0], 3), numpy.uint8)
HSVgem = numpy.zeros((color_resolution[1], color_resolution[0], 3), numpy.uint8)
white_image = numpy.zeros((color_resolution[1], color_resolution[0], 3), numpy.uint8)
#witte afbeelding om als masker te gebruiken
white_image[:] = [255, 255, 255]

#loop
while True:
    # opvragen van waarden van de sliders
    hmin = opencv.getTrackbarPos('Hmin', window_name)
    hmax = opencv.getTrackbarPos('Hmax', window_name)
    smin = opencv.getTrackbarPos('Smin', window_name)
    smax = opencv.getTrackbarPos('Smax', window_name)
    vmin = opencv.getTrackbarPos('Vmin', window_name)
    vmax = opencv.getTrackbarPos('Vmax', window_name)
    save = opencv.getTrackbarPos('save', window_name)

    # voorbeeldvensters inkleuren
    HSVmin[:] = [hmin, smin, vmin]
    HSVmax[:] = [hmax, smax, vmax]
    HSVgem[:] = [(hmin + hmax) / 2, (smin + smax) / 2, (vmin + vmax) / 2]

    # beelden omzetten van HSV naar BGR ruimte om te kunnen weergeven
    BGRmin = opencv.cvtColor(HSVmin, opencv.COLOR_HSV2BGR)
    BGRmax = opencv.cvtColor(HSVmax, opencv.COLOR_HSV2BGR)
    BGRgem = opencv.cvtColor(HSVgem, opencv.COLOR_HSV2BGR)

    #beelden ophalen
    color_image, depth_image = func.getframes(pipeline, align_object)

    #onderste en bovenste kleur definieren
    lower_color = numpy.array([hmin, smin, vmin])
    upper_color = numpy.array([hmax, smax, vmax])

    #masker van de bal opvragen
    mask=func.getmask(color_image,lower_color,upper_color)

    # masker toepassen op het camerabeeld
    res = opencv.bitwise_and(color_image, color_image, mask = mask)
    # zwart-wit voorstelling van het masker maken
    mask_bgr = opencv.bitwise_and(white_image, white_image, mask = mask)

    # alle beelden aan elkaar hechten en weergeven in het venster
    row1 = numpy.hstack((color_image, mask_bgr, res))
    row2 = numpy.hstack((BGRmin, BGRgem, BGRmax))
    img = numpy.vstack((row1, row2))

    #beelden laten zien
    func.showimage(img, window_name)

    #wanneer slider geactiveerd, verlaat loop
    if(save):
        break

hsvarray = numpy.array([hmin, hmax, smin, smax, vmin, vmax])
print(hsvarray)
numpy.save(datapath+'\hsv.npy', hsvarray)