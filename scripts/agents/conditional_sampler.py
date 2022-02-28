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
### Conditional Sampler ###
########################
class ConditionalSampler:

    def __init__(self, setting_dict):

        self.enable = setting_dict['if_use_CondSmpler']

        self.name = setting_dict['name']
        self.env_name = setting_dict['env_name']

        self.weight_lever = setting_dict['weight_lever']
        self.weight_img_attack_loss = setting_dict['weight_img_attack_loss']
        self.weight_reward_loss = setting_dict['weight_reward_loss']

        self.hidden_size = 128

        self.loss_model_given_s_n_a =  FeedforwardNN(setting_dict['N_STATE_DIM'] + setting_dict['N_ACT_DIM'], self.hidden_size, 1).to(DEVICE)
        self.loss_model_given_s =  FeedforwardNN(setting_dict['N_STATE_DIM'], self.hidden_size, 1).to(DEVICE)
        
        self.optim_loss_model_given_s_n_a = torch.optim.Adam(self.loss_model_given_s_n_a.parameters(), lr=setting_dict['lr_thompson'], betas=setting_dict['betas'])
        self.optim_loss_model_given_s = torch.optim.Adam(self.loss_model_given_s.parameters(), lr=setting_dict['lr_thompson'], betas=setting_dict['betas'])

    def update(self, s_arr, a_arr, r_arr, img_attack_loss_arr, attack_lever_arr):

        if self.enable:

            self.loss_model_given_s_n_a.train()
            self.loss_model_given_s.train()

            s_arr = torch.FloatTensor(s_arr).to(DEVICE)
            a_arr = torch.FloatTensor(a_arr).to(DEVICE)
            attack_lever_arr = torch.LongTensor(attack_lever_arr).to(DEVICE)

            # print('a_arr', a_arr.mean(), a_arr.min(), a_arr.max())
            # print('state_est', s_arr.mean(), s_arr.min(), s_arr.max())


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
            pred_loss_given_s_n_a = self.loss_model_given_s_n_a.forward(torch.cat((s_arr, a_arr),dim=1))
            pred_loss_given_s = self.loss_model_given_s.forward(s_arr)
            # ---------------------- optimize the model ----------------------
            loss1 = torch.mean((attack_loss - pred_loss_given_s_n_a)**2)
            loss2 = torch.mean((attack_loss - pred_loss_given_s)**2)

            # print('img_attack_loss_arr')
            # print(img_attack_loss_arr)

            # print('pred_loss_given_s_n_a')
            # print(pred_loss_given_s_n_a.mean(), pred_loss_given_s_n_a.min(), pred_loss_given_s_n_a.max())

            # print('pred_loss_given_s')
            # print(pred_loss_given_s)

            self.optim_loss_model_given_s_n_a.zero_grad()
            loss1.backward()
            self.optim_loss_model_given_s_n_a.step()        

            self.optim_loss_model_given_s.zero_grad()
            loss2.backward()
            self.optim_loss_model_given_s.step()

            loss1_val = loss1.item()
            loss2_val = loss2.item()


            # print(loss1.item(), loss2.item())        

            # '''
            # Calculate the regret!

            # - action use with loss what if I did ~action then just avg loss?
            # '''


            # '''
            # s_arr ([8, 32]) a_arr ([8, 4]) r_arr ([8]) img_attack_loss_arr ([8]) attack_lever_arr_one_hot ([8, 2])
            # '''
            # # ---------------------- forward the model ----------------------
            # x = torch.cat((s_arr, a_arr, img_attack_loss_arr.unsqueeze(1)),dim=1) # Get input
            # pred_reward_per_act = self.reward_model.forward(x)

            # # ---------------------- optimize reward estimator ----------------------
            # # Get performance, i.e., reward
            # #r_arr_adj = r_arr - (img_attack_loss_arr*attack_lever_arr - img_attack_loss_arr.mean())*self.weight_img_attack_loss
            # r_arr_adj = - (img_attack_loss_arr*attack_lever_arr - img_attack_loss_arr.mean())*self.weight_img_attack_loss
            # #r_arr_adj = r_arr #- img_attack_loss_arr #*attack_lever_arr*self.weight_img_attack_loss
            

            # print('img_attack_loss_arr', img_attack_loss_arr)
            # print('img_attack_loss_arr*attack_lever_arr', img_attack_loss_arr*attack_lever_arr)

            # print('pred_reward_per_act', pred_reward_per_act)
            # print('r_arr_adj', r_arr_adj)
            # print('attack_lever_arr', attack_lever_arr)
            # print('attack_lever_arr_one_hot', attack_lever_arr_one_hot)
            # print('torch.sum(attack_lever_arr_one_hot*pred_reward_per_act, dim=1)', torch.sum(attack_lever_arr_one_hot*pred_reward_per_act, dim=1))

            # print('errors', r_arr_adj - torch.sum(attack_lever_arr_one_hot*pred_reward_per_act, dim=1))



            # #loss = torch.mean((torch.sum(attack_lever_arr_one_hot*r_arr_adj, dim=1) - torch.sum(attack_lever_arr_one_hot*pred_reward_per_act, dim=1))**2)
            # loss = torch.mean((r_arr_adj - torch.sum(attack_lever_arr_one_hot*pred_reward_per_act, dim=1))**2)
            # self.reward_model_optimizer.zero_grad()
            # loss.backward()
            # self.reward_model_optimizer.step()
            # 
        else:
            loss1_val = 0
            loss2_val = 0        

        return loss1_val, loss2_val 


    def sample_lever_choice(self, state_est, action):

        if self.enable:

            with torch.no_grad():

                self.loss_model_given_s_n_a.train()
                self.loss_model_given_s.train()

                state_est = torch.FloatTensor(state_est).to(DEVICE)
                action = torch.FloatTensor(action).to(DEVICE)

                # print('action', action.mean(), action.min(), action.max())
                # print('state_est', state_est.mean(), state_est.min(), state_est.max())
                
                x1 = torch.cat((state_est, action)).unsqueeze(0).repeat(K_SAMPLES,1)

                # x1_tmp = torch.cat((state_est, action)).unsqueeze(0)
                # print('x1_tmp', x1_tmp.size())

                # tst =  self.loss_model_given_s_n_a.forward(x1_tmp)
                # print('tst', tst)
                
                
                # x1_tmp_repeat = x1_tmp.repeat(K_SAMPLES,1)
                # print('x1_tmp_repeat', x1_tmp_repeat.size())


                smpl_loss_given_s_n_a = self.loss_model_given_s_n_a.forward(x1)

                #print()

                x2 = state_est.unsqueeze(0).repeat(K_SAMPLES,1)
                smpl_loss_given_s = self.loss_model_given_s.forward(x2)

                mu0 = smpl_loss_given_s.mean().item()
                std0 = smpl_loss_given_s.std().item()
                s0 = np.random.normal(mu0, std0)

                mu1 = smpl_loss_given_s_n_a.mean().item()
                std1 = smpl_loss_given_s_n_a.std().item()
                s1 = np.random.normal(mu1, std1)
                

                if s1 < s0:
                    lever_value = 1 # when loss given state and action is better
                else:
                    lever_value = 0

            # print('s0', s0, 's1', s1)
            # print('mu0', mu0, 'mu1', mu1)
            # print('std0', std0, 's1', std1)

            # # state_est = torch.FloatTensor(state_est).to(DEVICE)
            # # # print(state_est)
            # # # print('state_est in sample_lever_choice', state_est.mean(), state_est.min(), state_est.max())
            # # action = torch.FloatTensor(action).to(DEVICE)
            # # # print('action in sample_lever_choice', action.mean(), action.min(), action.max())
            # # # img_attack_loss = torch.FloatTensor(np.array([img_attack_loss])).to(DEVICE)
            # # # print('img_attack_loss in sample_lever_choice', img_attack_loss.mean(), img_attack_loss.min(), img_attack_loss.max())


            # #x = torch.cat((state_est, action, img_attack_loss)) # Get input
            # # test = self.reward_model.forward(x)
            # # print('test', test)

            # # print('state_est', state_est.size(), 'action', action.size(), 'img_attack_loss', img_attack_loss.size())

            # ### Sample estimated rewards ###
            # x = torch.cat((state_est, action, img_attack_loss)).unsqueeze(0).repeat(K_SAMPLES,1)
            # #x =(state_est).unsqueeze(0).repeat(K_SAMPLES,1)
            # sample_reward_per_act = self.reward_model.forward(x)

            # print('sample_reward_per_act', sample_reward_per_act)


            # # print('sample_reward_per_act[:,0].mean()', sample_reward_per_act[:,0].mean())

            # mu0 = sample_reward_per_act[:,0].mean().item()
            # std0 = sample_reward_per_act[:,0].std().item()
            # s0 = np.random.normal(mu0, std0)

            # mu1 = sample_reward_per_act[:,1].mean().item()
            # std1 = sample_reward_per_act[:,1].std().item()
            # s1 = np.random.normal(mu1, std1)

            # # print('s0', s0, 's1', s1)
            # # print('mu0', mu0, 'mu1', mu1)
            # # print('std0', std0, 's1', std1)

            # if s0 < s1:
            #     lever_value = 1
            # else:
            #     lever_value = 0

            # if np.random.uniform() < 0.05:

            #     lever_value = np.random.choice(2)


            
            # # max_reward_act0 = torch.max(sample_reward_per_act[:,0])
            # # max_reward_act1 = torch.max(sample_reward_per_act[:,1])

            # # if max_reward_act0 < max_reward_act1:
            # #     lever_value = 1
            # # else:
            # #     lever_value = 0

            # # print('max_reward_act0:', max_reward_act0)
            # # print('max_reward_act1:', max_reward_act1)
            # # print('lever_value:', lever_value)

        else:
            lever_value = 1

        return lever_value, mu0, std0, mu1, std1

    def save_the_model(self):

        if self.enable:

            if not os.path.exists('save/'+self.env_name+'/save/ts/'):
                os.makedirs('save/'+self.env_name+'/save/ts/')
            f_name = self.name + '_model_given_s_n_a_param_' + '_model.pth'
            torch.save(self.loss_model_given_s_n_a.state_dict(), 'save/'+self.env_name+'/save/ts/'+f_name)
            f_name = self.name + '_model_given_s_param_' + '_model.pth'
            torch.save(self.loss_model_given_s.state_dict(), 'save/'+self.env_name+'/save/ts/'+f_name)
        
        

    def load_the_model(self):

        if self.enable:

            f_name = self.name + '_model_given_s_n_a_param_'  + '_model.pth'
            self.loss_model_given_s_n_a.load_state_dict(torch.load('save/'+self.env_name+'/save/ts/'+f_name))
            f_name = self.name + '_model_given_s_param_' + '_model.pth'
            self.loss_model_given_s.load_state_dict(torch.load('save/'+self.env_name+'/save/ts/'+f_name))
            


