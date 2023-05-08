import serial  # pip install pyserial
import time
device = 'COM6'
Serial = serial.Serial(device, 115200, timeout=.1)
time.sleep(2)
def get_acc():
    while not Serial.in_waiting:
        pass
    #return (Serial.read(2), Serial.read(2))
    return (int.from_bytes(Serial.read(2), "little", signed=True), int.from_bytes(Serial.read(2), "little", signed=True))

while True:
    #print(type(get_acc()[0]))
    print(get_acc())
