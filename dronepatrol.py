import cv2
import os
import sys
import math
import time

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

from dronekit import connect, VehicleMode, mavutil
from dronekit.mavlink import MAVConnection

def set_servo(servo_number, pwm_value):
    pwm_value_int = int(pwm_value)
    msg = vehicle.message_factory.command_long_encode(0,0,mavutil.mavlink.MAV_CMD_DO_SET_SERVO,0,servo_number, pwm_value_int,0,0,0,0,0)
    vehicle.send_mavlink(msg)

def my_new_fix_targets(message):
    pass

connection_string = "/dev/ttyACM0"
baud_rate = 115200
print(">>>> Connecting with the UAV <<<")
vehicle = connect(connection_string,baud_rate, wait_ready=True)     #- wait_ready flag hold the program untill all the parameters are been read (=, not .)
udp_conn = MAVConnection('udpin:192.168.0.22:15667', source_system=1)
vehicle._handler.pipe(udp_conn)
udp_conn.master.mav.srcComponent = 1  # needed to make QGroundControl work!
udp_conn.start()

udp_conn.fix_targets = my_new_fix_targets
#msg = vehicle.message_factory.statustext_encode(1,b'Alert Fire at : ' + str( vehicle.location.global_frame).encode())
msg = vehicle.message_factory.statustext_encode(1,b'Alert Fire: LocationGlobal:lat=48.7965,lon=2.2097')
time.sleep(10)
udp_conn.master.mav.send(msg)
pwm_value_int = int(1890)
msg1 = vehicle.message_factory.command_long_encode(0,0,mavutil.mavlink.MAV_CMD_DO_SET_SERVO,0,13, pwm_value_int,0,0,0,0,0)
udp_conn.master.mav.send(msg1)
#set_servo(13,1980)
#set_servo(13,1500)
while not vehicle.is_armable:
    print("waiting to be armable")
    time.sleep(1)

print("Arming motors")
vehicle.mode = VehicleMode("AUTO")
vehicle.armed = True

while not vehicle.armed: time.sleep(1)


def construct_firenet (x,y):

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    network = fully_connected(network, 4096, activation='tanh')
    network = fully_connected(network, 2, activation='softmax')

    # constuct final model

    model = tflearn.DNN(network, checkpoint_path='firenet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model

################################################################################

if __name__ == '__main__':

################################################################################

    # construct and display model

    model = construct_firenet (224, 224)
    print("Constructed FireNet ...")

    model.load(os.path.join("model", "fire"),weights_only=True)
    print("Loaded CNN network weights ...")

################################################################################

    # network input sizes

    rows = 224
    cols = 224

    # display and loop settings

    windowName = "HoverGames Challenge 1: Fight Fire with Flyers with NXP";
    keepProcessing = True;


    #- Read the actual position
    print('Position: %s'% vehicle.location.global_relative_frame)

    #- Read the actual attitude roll, pitch, yaw
    print('Attitude: %s'% vehicle.attitude)

    #- What is the actual flight mode?    Note: this is settable
    print('Mode: %s'% vehicle.mode.name)

    

    # load video file from first command line argument

    video = cv2.VideoCapture(0)
    print("Loaded video ...")

    # create window

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);

    # get video properties

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_time = round(1000/fps);

    while (keepProcessing):

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount();

        # get video frame from file, handle end of file

        ret, frame = video.read()
        if not ret:
            print("... end of video file reached");
            break;

        # re-size image to network input size and perform prediction

        small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)
        output = model.predict([small_frame])

        # label image based on prediction

        if round(output[0][0]) == 1:
            cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
            cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),7,cv2.LINE_AA);
        else:
            cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
            cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),7,cv2.LINE_AA);
            msg2 = vehicle.message_factory.statustext_encode(1,b'fire')
            vehicle.send_mavlink(msg2)

        # stop the timer and convert to ms. (to see how long processing and display takes)

        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

        # image display and key handling

        cv2.imshow(windowName, frame);

        # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

        key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF;
        if (key == ord('x')):
            keepProcessing = False;
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);