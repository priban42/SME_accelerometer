import cv2
import numpy as np
from ComputerVision import Computer_vision
import serial  # pip install pyserial
import time
device = 'COM6'
Serial = serial.Serial(device, 115200, timeout=.1)
Serial.flushInput()
time.sleep(2)

vid = cv2.VideoCapture(0)
computer_vision = Computer_vision()

def get_acc():
    while not Serial.in_waiting:
        pass
    while int.from_bytes(Serial.read(1)) != 0:
        pass
    #return (Serial.read(2), Serial.read(2))
    b1 = int.from_bytes(Serial.read(1), "big", signed=False)
    b2 = int.from_bytes(Serial.read(1), "big", signed=False)
    #print(b1, b2, "..")
    acc = np.array([b1, b2])
    Serial.flushInput()
    return (acc - np.array([127, 127]))*3

while (True):
    ret, frame = vid.read()
    computer_vision.update_image(frame)
    #if computer_vision.display_bgr_image():
    computer_vision.get_color_position("green")
    center = computer_vision.get_color_position("pink")
    origin = computer_vision.get_color_position("green")
    #print(np.array(get_acc()))
    vect = get_acc()# np.array([20, 20])
    if computer_vision.display_acc(center, origin, vect, "green", "pink"):
    #if computer_vision.display_contours("green", "pink"):
        break

    #cv2.imshow("BGR", frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()