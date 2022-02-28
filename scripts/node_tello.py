#!/usr/bin/env python3
import rospy, cv2




from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from djitellopy import Tello   #https://pypi.org/project/djitellopy2/       https://djitellopy.readthedocs.io/en/latest/


S = 10
S_CTL = 10

ENABLE_TAKE_OFF = True


### Global Objects and Variables ###
tello_agent = Tello()
tello_agent.connect()
tello_agent.streamon()
frame_read_object = tello_agent.get_frame_read()

land_cmd = False
takeoff_cmd = False


### ROS Subscriber Callback ###
KEY_CMD_RECEIVED = None
def fnc_callback(msg):
    global KEY_CMD_RECEIVED
    KEY_CMD_RECEIVED = msg

CTL_CMD_RECEIVED = None
def fnc_callback1(msg):
    global CTL_CMD_RECEIVED
    CTL_CMD_RECEIVED = msg
    

CTL_SWITCH = 0


if __name__=='__main__':

    # rosnode node initialization
    rospy.init_node('tello_node')

    # subscriber init.
    sub_key_cmd         = rospy.Subscriber('/key_teleop/vel_cmd_body_frame', Twist, fnc_callback)
    sub_vel_cmd_control = rospy.Subscriber('/controller_node/vel_cmd', Vector3, fnc_callback1)
    # publishers init.
    pub_camera_frame = rospy.Publisher('/tello_node/camera_frame', Image, queue_size=10)
    pub_state_values = rospy.Publisher('/tello_node/state_values', Float32MultiArray, queue_size=10)

    # msg init. the msg is to send out state value array.
    msg_mat = Float32MultiArray()
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim[0].label = "height"
    msg_mat.layout.dim[1].label = "width"

    # a bridge from cv2 image to ROS image
    mybridge = CvBridge()

    # Running rate at 10 Hz
    rate=rospy.Rate(10)

    ##############################
    ### Instructions in a loop ###
    ##############################
    while not rospy.is_shutdown():

        t_start = rospy.get_rostime()

        if KEY_CMD_RECEIVED is not None:

            ###########################################
            ### Execute the command from key_teleop ###                               
            ###########################################

            ### Landing ###
            if KEY_CMD_RECEIVED.angular.x < -0.5:
                tello_agent.land()
                print('landing finished')
                break

            ### Taking off ###
            elif KEY_CMD_RECEIVED.angular.x > 0.5:
                if ENABLE_TAKE_OFF:
                    tello_agent.takeoff()
                    print('take off finished')
                else:
                    print('I disabled it!')

            ### Engaging Controller ###
            elif KEY_CMD_RECEIVED.angular.y < -0.5:
                CTL_SWITCH = 0
                print('Engaging  -> ', 'CTL_SWITCH:', CTL_SWITCH)

            ### DISengaging Controller ###
            elif KEY_CMD_RECEIVED.angular.y > 0.5:
                CTL_SWITCH = 1
                print('DISengaging  -> ', 'CTL_SWITCH:', CTL_SWITCH)

            else:
                ### Send command ###
                forward_backward_velocity = KEY_CMD_RECEIVED.linear.x
                left_right_velocity       = KEY_CMD_RECEIVED.linear.y
                up_down_velocity          = KEY_CMD_RECEIVED.linear.z
                yaw_velocity              = KEY_CMD_RECEIVED.angular.z


                if CTL_CMD_RECEIVED is not None:
                    control_cmd_V_right = CTL_CMD_RECEIVED.x
                    control_cmd_V_up = CTL_CMD_RECEIVED.y
                    control_cmd_V_forward = CTL_CMD_RECEIVED.z

                else:
                    control_cmd_V_right = 0
                    control_cmd_V_up = 0
                    control_cmd_V_forward = 0


                print('CTL_SWITCH:', CTL_SWITCH)
                

                left_right_velocity       = CTL_SWITCH*S_CTL*control_cmd_V_right*1.0   + S*left_right_velocity
                up_down_velocity          = CTL_SWITCH*S_CTL*control_cmd_V_up*0.8      + S*up_down_velocity
                forward_backward_velocity = CTL_SWITCH*S_CTL*control_cmd_V_forward*0.4 + S*forward_backward_velocity
                #forward_backward_velocity = S*forward_backward_velocity

                print('control_cmd_V_right', left_right_velocity)
                print('control_cmd_V_up', up_down_velocity)
                print('control_cmd_V_forward', forward_backward_velocity)



                tello_agent.send_rc_control(int(left_right_velocity), int(forward_backward_velocity), int(up_down_velocity), int(yaw_velocity))

                


        ###########################
        ### Publish Sensor Data ###                               
        ###########################

        ### Get image ###
        cv2_img = cv2.resize(frame_read_object.frame, (448,448))
        img = mybridge.cv2_to_imgmsg(cv2_img)        

        ### Get state ###
        state = tello_agent.get_current_state()

        '''
        {'pitch': 0, 'roll': -1, 'yaw': 0,
         'vgx': 0, 'vgy': 0, 'vgz': 0,
         'templ': 54, 'temph': 57,
         'tof': 10, 'h': 0,
         'bat': 100, 'baro': 1409.32, 'time': 0,
         'agx': 0.0, 'agy': 8.0, 'agz': -995.0}
        '''

        state_list = [float(state['pitch']), float(state['roll']), float(state['yaw'])]
        state_list+= [float(state['vgx']), float(state['vgy']), float(state['vgz'])]
        state_list+= [float(state['templ']), float(state['temph']), float(state['tof'])]
        state_list+= [float(state['h']), float(state['bat']), float(state['baro']), float(state['time'])]
        state_list+= [float(state['agx']), float(state['agy']), float(state['agz'])]


        # Continued imitialzation of the msg. here, its size is varied. And the prediction result is saved as float array.
        msg_mat.layout.dim[0].size = 1
        msg_mat.layout.dim[1].size = 16
        msg_mat.layout.dim[0].stride = 1*16
        msg_mat.layout.dim[1].stride = 16
        msg_mat.layout.data_offset = 0
        msg_mat.data = state_list
        


        ### Publish Image and State ###
        pub_camera_frame.publish(img)
        pub_state_values.publish(msg_mat)


        # Sleeping to meet the frequency
        rate.sleep()
        delta_time = rospy.get_rostime() - t_start
        #print('----------------a loop----------------', delta_time.to_sec())


    tello_agent.end()