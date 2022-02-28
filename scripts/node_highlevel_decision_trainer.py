#!/usr/bin/env python3

# Train the state estimator (dynamics autoencoder) and the RL agent that generates target (or action)

import rospy
from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import MultiArrayDimension      # See http://docs.ros.org/api/std_msgs/html/msg/MultiArrayLayout.html
from sensor_msgs.msg import Image
import numpy as np
import cv2

from setting_params import SETTING, FREQ_HIGH_LEVEL, DEVICE
from agents.dynamic_auto_encoder import DynamicAutoEncoderAgent
from agents.rl_agent import DDPGAgent

if SETTING['if_use_CondSmpler']:
    from agents.conditional_sampler import ConditionalSampler as BinaryDecisionMaker
    print('Conditional Sampler Initiated!')
elif SETTING['if_use_TS']:
    from agents.thompson_sampler import ThompsonSampler as BinaryDecisionMaker
    print('Thompson Sampler Initiated!')
else:
    print('Unknown decision maker!')
    from agents.thompson_sampler import ThompsonSampler as BinaryDecisionMaker


from memory import SingleTrajectoryBuffer, TransitionBuffer 

IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

TRANSITION_EST_RECEIVED = None
def fnc_img_callback1(msg):
    global TRANSITION_EST_RECEIVED
    TRANSITION_EST_RECEIVED = msg

#<----------- TS -------------
ATTACK_LEVER_RECEIVED = None
def fnc_img_callback2(msg):
    global ATTACK_LEVER_RECEIVED
    ATTACK_LEVER_RECEIVED = msg

LOSS_IMAGE_ATTACK_LOSS_RECEIVED = None
def fnc_loss3_callback(msg):
    global LOSS_IMAGE_ATTACK_LOSS_RECEIVED
    LOSS_IMAGE_ATTACK_LOSS_RECEIVED = msg
#<----------- TS -------------


if __name__ == '__main__':
    
    # rosnode node initialization
    rospy.init_node('high_level_decision_trainer')

    # subscriber init.
    sub_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_img_callback)
    sub_state_transition_observation = rospy.Subscriber('/decision_maker_node/state_est_transition', Float32MultiArray, fnc_img_callback1)
    
    #<----------- TS -------------
    sub_attack_lever = rospy.Subscriber('/decision_maker_node/attack_lever', Float32, fnc_img_callback2)
    sub_loss_image_attack = rospy.Subscriber('/attack_generator_node/attack_loss', Float32, fnc_loss3_callback)
    # sub_attack_loss_transition = rospy.Subscriber('/memory_node/attack_loss', Float32MultiArray, fnc_transition_attack_loss_callback)
    # sub_attack_lever_transition = rospy.Subscriber('/memory_node/attack_lever', Float32MultiArray, fnc_transition_attack_lever_callback)
    #<----------- TS -------------


    # publishers init.
    pub_loss_monitor = rospy.Publisher('/decision_trainer_node/loss_monitor', Float32MultiArray, queue_size=3)   # publisher1 initialization.

    # Running rate
    rate=rospy.Rate(FREQ_HIGH_LEVEL)

    # Training agents init
    SETTING['name'] = rospy.get_param('name')
    state_estimator = DynamicAutoEncoderAgent(SETTING, train=True)
    rl_agent = DDPGAgent(SETTING)

    #<----------- TS -------------
    conditional_sampler = BinaryDecisionMaker(SETTING)
    #<----------- TS -------------

    # Memory init
    single_trajectory_memory = SingleTrajectoryBuffer(SETTING['N_SingleTrajectoryBuffer'])
    transition_memory = TransitionBuffer(SETTING['N_TransitionBuffer'])
    #<----------- TS -------------
    conditional_smpl_memory = TransitionBuffer(SETTING['N_TransitionBuffer'])
    #<----------- TS -------------
    

    # msg init. the msg is to send out numpy array.
    msg_mat = Float32MultiArray()
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim.append(MultiArrayDimension())
    msg_mat.layout.dim[0].label = "height"
    msg_mat.layout.dim[1].label = "width"

    ##############################
    ### Instructions in a loop ###
    ##############################
    n_iteration = 0
    while not rospy.is_shutdown():

        if IMAGE_RECEIVED is not None and TRANSITION_EST_RECEIVED is not None and LOSS_IMAGE_ATTACK_LOSS_RECEIVED is not None and ATTACK_LEVER_RECEIVED:
            n_iteration += 1

            ### Add samples to the buffers ###
            # unpack image
            np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
            np_im = np.array(np_im)
            np_im = cv2.resize(np_im, SETTING['encoder_image_size'], interpolation = cv2.INTER_AREA)

            # unpack state
            height = TRANSITION_EST_RECEIVED.layout.dim[0].size
            width = TRANSITION_EST_RECEIVED.layout.dim[1].size
            np_transition = np.array(TRANSITION_EST_RECEIVED.data).reshape((height, width))

            # pack state transition
            prev_np_state_estimate = np_transition[0]
            action = np_transition[1][:SETTING['N_ACT_DIM']]
            reward = np_transition[1][-1]
            done = np_transition[1][-2]
            np_state_estimate = np_transition[2]


            #print('prev_np_state_estimate in trainer', np_state_estimate)

            # add data into memory
            single_trajectory_memory.add(np_im, action, prev_np_state_estimate)
            transition_memory.add(prev_np_state_estimate, action, reward, np_state_estimate, done)

            #<----------- TS -------------
            image_attack_loss = LOSS_IMAGE_ATTACK_LOSS_RECEIVED.data
            attack_lever = ATTACK_LEVER_RECEIVED.data
            conditional_smpl_memory.add(np_state_estimate, action, reward, image_attack_loss, attack_lever)
            #print('thompson_smpl_memory.add', np_state_estimate)
            #<----------- TS -------------

            ####################################################
            ## CAL THE LOSS FUNCTION & A STEP OF GRAD DESCENT ##
            ####################################################
            if single_trajectory_memory.len > SETTING['N_WINDOW'] and transition_memory.len > SETTING['N_WINDOW']:

                # sample minibach
                batch_obs_img_stream, batch_tgt_stream, batch_state_est_stream = single_trajectory_memory.sample(SETTING['N_WINDOW'])
                s_arr, a_arr, r_arr, s1_arr, done_arr = transition_memory.sample(SETTING['N_MINIBATCH_DDPG'])

                # update the models
                loss_sys_id = state_estimator.update(batch_obs_img_stream, batch_state_est_stream, batch_tgt_stream)
                loss_actor, loss_critic = rl_agent.update(s_arr, a_arr, r_arr, s1_arr, done_arr)

                #<----------- TS -------------
                ######################################
                ### Thompson Sampling Model Update ###
                ######################################


                s_arr, a_arr, r_arr, img_attack_loss_arr, attack_lever_arr = conditional_smpl_memory.sample(SETTING['N_MINIBATCH_DDPG']*4)
                #print('thompson_smpl_memory.sample', s_arr)
                loss_condtionalSmpl1, loss_condtionalSmpl2 = conditional_sampler.update(s_arr, a_arr, r_arr, img_attack_loss_arr, attack_lever_arr)

                #print('loss_thompson', loss_thompson)

                # if n_iteration > 2000:
                #     loss_actor, loss_critic = rl_agent.update(s_arr, a_arr, r_arr, s1_arr, done_arr)
                # else:
                #     loss_actor = 0
                #     loss_critic = 0

                # pack up loss values
                loss_monitor_np = np.array([[loss_sys_id, loss_actor, loss_critic, loss_condtionalSmpl1, loss_condtionalSmpl2]])
                msg_mat.layout.dim[0].size = loss_monitor_np.shape[0]
                msg_mat.layout.dim[1].size = loss_monitor_np.shape[1]
                msg_mat.layout.dim[0].stride = loss_monitor_np.shape[0]*loss_monitor_np.shape[1]
                msg_mat.layout.dim[1].stride = loss_monitor_np.shape[1]
                msg_mat.layout.data_offset = 0
                msg_mat.data = loss_monitor_np.flatten().tolist()
                pub_loss_monitor.publish(msg_mat)

        if n_iteration % (FREQ_HIGH_LEVEL+1) ==0:
            try:
                state_estimator.save_the_model()
                rl_agent.save_the_model()
                conditional_sampler.save_the_model()
            except:
                print('in high_level_decision_trainer, model saving failed!')

        
        try:
            experiment_done_done = rospy.get_param('experiment_done')
        except:
            experiment_done_done = False
        if experiment_done_done and n_iteration > FREQ_HIGH_LEVEL*3:
            rospy.signal_shutdown('Finished 100 Episodes!')


            
            
        rate.sleep()
