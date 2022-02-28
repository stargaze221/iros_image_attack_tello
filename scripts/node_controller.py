#!/usr/bin/env python3
import numpy as np
import rospy

from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from geometry_msgs.msg import Vector3


### ROS Subscriber Callback ###
STATE_ARRAY_RECEIVED = None
def fnc_callback(msg):
    global STATE_ARRAY_RECEIVED
    STATE_ARRAY_RECEIVED = msg

PREDICTION_ARRAY_RECEIVED = None
def fnc_callback1(msg):
    global PREDICTION_ARRAY_RECEIVED
    PREDICTION_ARRAY_RECEIVED = msg

P_gain = 1.0
TARGET_CLASS_INDEX = 39


if __name__=='__main__':

    # rosnode node initialization
    rospy.init_node('controller_node')

    # subscriber init.
    sub  = rospy.Subscriber('/tello_node/state_values', Float32MultiArray, fnc_callback)
    sub1 = rospy.Subscriber('/yolo_node/yolo_predictions', Float32MultiArray, fnc_callback1)


    # publishers init.
    pub_vel_est = rospy.Publisher('/controller_node/vel_est_rcvd', Vector3, queue_size=1)
    pub_body_angle = rospy.Publisher('/controller_node/body_angle_rcvd', Vector3, queue_size=1)
    pub_tgt_box = rospy.Publisher('/controller_node/tgt_box_rcvd', Vector3, queue_size=1)
    pub_vel_cmd = rospy.Publisher('/controller_node/vel_cmd', Vector3, queue_size=1)

    # Running rate at 10 Hz
    rate=rospy.Rate(5)

    vel_est    = Vector3()
    body_angle = Vector3()
    tgt_box    = Vector3()
    vel_cmd_tracking = Vector3()


    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():

        if STATE_ARRAY_RECEIVED is not None:

            height = STATE_ARRAY_RECEIVED.layout.dim[0].size
            width = STATE_ARRAY_RECEIVED.layout.dim[1].size

            np_state = np.array(STATE_ARRAY_RECEIVED.data).reshape((height, width))

            pitch, roll, yaw = (np_state[0][0], np_state[0][1], np_state[0][2])
            vgx, vgy, vgz = (np_state[0][3], np_state[0][4], np_state[0][5])
            templ, temph = (np_state[0][6], np_state[0][7])
            tof, h = (np_state[0][8], np_state[0][9])
            bat, baro, time = (np_state[0][10], np_state[0][11], np_state[0][12])
            agx, agy, agz = (np_state[0][13], np_state[0][14], np_state[0][15])

            vel_est.x, vel_est.y, vel_est.z = (vgx, vgy, vgz)
            body_angle.x, body_angle.y, body_angle.z = (pitch, roll, yaw)

            if PREDICTION_ARRAY_RECEIVED is not None:
                height = PREDICTION_ARRAY_RECEIVED.layout.dim[0].size
                width = PREDICTION_ARRAY_RECEIVED.layout.dim[1].size
                np_prediction = np.array(PREDICTION_ARRAY_RECEIVED.data).reshape((height, width))

                #                   x1           y1           x2           y2   confidence        class
                # tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
                #         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
                #         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])


                if np_prediction.shape[0] == 1 and np_prediction.shape[1] == 1:
                    print("No detection")
                     
                else:
                    tgt_boxes = np_prediction[np.where(np_prediction[:,5]==TARGET_CLASS_INDEX)]
                    #print('bottle')
                    x_ctr = (tgt_boxes[:,0] + tgt_boxes[:,2])/2/100
                    y_ctr = (tgt_boxes[:,1] + tgt_boxes[:,3])/2/100
                    size  = (tgt_boxes[:,2] - tgt_boxes[:,0])*(tgt_boxes[:,3] - tgt_boxes[:,1])/1000

                    # -------------------------------#
                    #|             y = 0
                    #|
                    #|
                    #| x = 0      (2.4, 3.5)      x = 5
                    #|
                    #|
                    #|             y = 5
                    # -------------------------------#

                    # size [10, 100]   target 30
                    CTR_X_POS = 2.4
                    CTR_Y_POS = 3.0
                    AREA_TGT = 7

                    print('x_ctr', x_ctr, 'y_ctr', y_ctr, 'size', size)

                    if len(tgt_boxes) > 0:
                        # Take the most believable bounding box if there are multiple of them.
                        tgt_box.x, tgt_box.y, tgt_box.z = (float(x_ctr[0]), float(y_ctr[0]), float(size[0]))

                        ### Now, make the control signal ###
                        error_x = tgt_box.x - CTR_X_POS
                        error_y = tgt_box.y - CTR_Y_POS
                        error_z = (tgt_box.z)**0.5 - (AREA_TGT)**0.5

                        cmd_vx = P_gain * error_x    # side move (left or right)
                        cmd_vy = -P_gain*0.8* error_y   # vertical move (up or down)
                        cmd_vz = -P_gain * error_z   # front move (front or backward)

                        vel_cmd_tracking.x = cmd_vx  # if target is at the right then generate positive cmd_vx
                        vel_cmd_tracking.y = cmd_vy  # if target is at the above then generate positive cmd_vy
                        vel_cmd_tracking.z = cmd_vz  # if target is small then generate positive cmd_vz

                    else:
                        vel_cmd_tracking.x = 0
                        vel_cmd_tracking.y = 0
                        vel_cmd_tracking.z = 0


            else:
                vel_cmd_tracking.x = 0
                vel_cmd_tracking.y = 0
                vel_cmd_tracking.z = 0


        ### Publish Image and State ###
        pub_vel_est.publish(vel_est)
        pub_body_angle.publish(body_angle)
        pub_tgt_box.publish(tgt_box)
        pub_vel_cmd.publish(vel_cmd_tracking)

        rate.sleep()