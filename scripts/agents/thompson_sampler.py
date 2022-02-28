#from Tello_Image.model import DetectorNetwork, DynamicAutoEncoderNetwork, ImageAttackNetwork, Critic, Actor
from nn_networks.reward_est_nn import FeedforwardNN



import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from setting_params import DEVICE, N_ACT_DIM, N_STATE_DIM

import os

K_SAMPLES = 8


########################
### Thompson Sampler ###
########################
class ThompsonSampler:

    def __init__(self, setting_dict):

        self.enable = setting_dict['if_use_TS']

        self.name = setting_dict['name']
        self.env_name = setting_dict['env_name']

        self.weight_lever = setting_dict['weight_lever']
        self.weight_img_attack_loss = setting_dict['weight_img_attack_loss']
        self.weight_reward_loss = setting_dict['weight_reward_loss']

        self.hidden_size = 128

        self.loss_model_given_state_n_act_n_bin =  FeedforwardNN(setting_dict['N_STATE_DIM'] + setting_dict['N_ACT_DIM']+1, self.hidden_size, 1).to(DEVICE)
        self.optim_loss_model_given_state_n_act_n_bin = torch.optim.Adam(self.loss_model_given_state_n_act_n_bin.parameters(), lr=setting_dict['lr_thompson'], betas=setting_dict['betas'])
        

    def update(self, s_arr, a_arr, r_arr, img_attack_loss_arr, attack_lever_arr):

        if self.enable:

            self.loss_model_given_state_n_act_n_bin.train()

            s_arr = torch.FloatTensor(s_arr).to(DEVICE)
            a_arr = torch.FloatTensor(a_arr).to(DEVICE)
            attack_lever_arr = torch.FloatTensor(attack_lever_arr).to(DEVICE)
            
            r_arr = torch.FloatTensor(r_arr).to(DEVICE)
            img_attack_loss_arr = torch.FloatTensor(img_attack_loss_arr).to(DEVICE)

            attack_loss  = self.weight_img_attack_loss * img_attack_loss_arr  + self.weight_lever * attack_lever_arr  - self.weight_reward_loss*r_arr

            

            '''
            s_arr ([8, 32]) a_arr ([8, 4]) r_arr ([8]) img_attack_loss_arr ([8]) attack_lever_arr_one_hot ([8, 2])
            '''

            '''
            We want to compare the following:

            - the particular loss function given state, action, i.e., pred_loss1
            - the mean loss function given state, i.e., pred_loss2

            if pred_loss1 < pred_loss2:
                use the attack
            else:
                no attack

            OK, Let me try this!
            '''
            # ---------------------- forward the model ----------------------
            pred_loss_given_s_n_a_n_bin = self.loss_model_given_state_n_act_n_bin.forward(torch.cat((s_arr, a_arr, attack_lever_arr.unsqueeze(1)),dim=1))
            # ---------------------- optimize the model ----------------------
            loss = torch.mean((attack_loss - pred_loss_given_s_n_a_n_bin.squeeze())**2)
            
            self.optim_loss_model_given_state_n_act_n_bin.zero_grad()
            loss.backward()
            self.optim_loss_model_given_state_n_act_n_bin.step()        
            
            loss1_val = loss.item()
            loss2_val = 0

        else:
            loss1_val = 0
            loss2_val = 0        

        return loss1_val, loss2_val 


    def sample_lever_choice(self, state_est, action):

        if self.enable:
            with torch.no_grad():
                self.loss_model_given_state_n_act_n_bin.train()

                state_est = torch.FloatTensor(state_est).to(DEVICE)
                action = torch.FloatTensor(action).to(DEVICE)
                x1 = torch.cat((state_est, action)).unsqueeze(0).repeat(K_SAMPLES,1)
                ones = torch.ones(K_SAMPLES).unsqueeze(1).to(DEVICE)
                x1 = torch.hstack((x1, ones))
                smpl_loss_given_s_n_a_bin_1 = self.loss_model_given_state_n_act_n_bin.forward(x1)



                x0 = torch.cat((state_est, action)).unsqueeze(0).repeat(K_SAMPLES,1)
                zeros = torch.zeros(K_SAMPLES).unsqueeze(1).to(DEVICE)
                x0 = torch.hstack((x0, zeros))
                smpl_loss_given_s_n_a_bin_0 = self.loss_model_given_state_n_act_n_bin.forward(x0)

                mu0 = smpl_loss_given_s_n_a_bin_0.mean().item()
                std0 = smpl_loss_given_s_n_a_bin_0.std().item()
                s0 = np.random.normal(mu0, std0)

                mu1 = smpl_loss_given_s_n_a_bin_1.mean().item()
                std1 = smpl_loss_given_s_n_a_bin_1.std().item()
                s1 = np.random.normal(mu1, std1)


                if s1 < s0:
                    lever_value = 1 # when loss given state and action is better
                else:
                    lever_value = 0
        
        else:
            lever_value = 1

        return lever_value

        

    def save_the_model(self):

        if self.enable:

            if not os.path.exists('save/'+self.env_name+'/save/ts/'):
                os.makedirs('save/'+self.env_name+'/save/ts/')
            f_name = self.name + '_model_given_state_n_act_n_bin_param_' + '_model.pth'
            torch.save(self.loss_model_given_state_n_act_n_bin.state_dict(), 'save/'+self.env_name+'/save/ts/'+f_name)
            # f_name = self.name + '_model_given_s_param_' + '_model.pth'
            # torch.save(self.loss_model_given_s.state_dict(), 'save/'+self.env_name+'/save/ts/'+f_name)
        
        

    def load_the_model(self):

        if self.enable:

            f_name = self.name + '_model_given_state_n_act_n_bin_param_'  + '_model.pth'
            self.loss_model_given_state_n_act_n_bin.load_state_dict(torch.load('save/'+self.env_name+'/save/ts/'+f_name))
            # f_name = self.name + '_model_given_s_param_' + '_model.pth'
            # self.loss_model_given_s.load_state_dict(torch.load('save/'+self.env_name+'/save/ts/'+f_name))
            



