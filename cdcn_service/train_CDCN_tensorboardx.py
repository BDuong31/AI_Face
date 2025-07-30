from __future__ import print_function, division
import torch
import matplotlib as mpl
mpl.use('TkAgg')
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from models.CDCNs import Conv2d_cd, CDCNpp

from Loadtemporal_BinaryMask_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Loadtemporal_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from utils import AvgrageMeter, accuracy, performances
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
# Dataset root
image_dir = os.path.join(base_dir, 'datamain')
train_list = os.path.join(image_dir, 'protocol', 'train_list.txt')
val_list = os.path.join(image_dir, 'protocol', 'val_list.txt')

def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''
    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float64)).float().cpu()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)
        
        criterion_MSE = nn.MSELoss().cpu()
    
        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)
    
        return loss


# main function
def train_test():
    # --- THÊM VÀO: Khởi tạo TensorBoardX SummaryWriter ---
    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'w')
    writer = SummaryWriter(log_dir=args.log) # <-- THÊM VÀO: Khởi tạo writer
    # ----------------------------------------------------

    echo_batches = args.echo_batches

    print("Oulu-NPU, P1:\n ")
    log_file.write('Oulu-NPU, P1:\n ')
    log_file.flush()

    # load the network
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')
        # Logic tải model finetune sẽ ở đây
    else:
        print('train from scratch!\n')
        log_file.write('train from scratch!\n')
        log_file.flush()
         
        model = CDCNpp(basic_conv=Conv2d_cd, theta=args.theta)
        model = model.cpu()

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    print(model) 
    
    criterion_absolute_loss = nn.MSELoss().cpu()
    criterion_contrastive_loss = Contrast_depth_loss().cpu() 

    # --- THÊM VÀO: Biến đếm bước toàn cục để ghi log chi tiết hơn ---
    global_step = 0
    # ------------------------------------------------------------
    
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        # --- THÊM VÀO: Ghi lại learning rate mỗi epoch ---
        writer.add_scalar('Train/Learning_Rate', lr, epoch)
        # -------------------------------------------
        
        loss_absolute = AvgrageMeter()
        loss_contra =  AvgrageMeter()
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        train_data = Spoofing_train(train_list, image_dir, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)

        for i, sample_batched in enumerate(dataloader_train):
            inputs, binary_mask, spoof_label = sample_batched['image_x'].cpu(), sample_batched['binary_mask'].cpu(), sample_batched['spoofing_label'].cpu() 

            optimizer.zero_grad()
            
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)
            
            absolute_loss = criterion_absolute_loss(map_x, binary_mask)
            contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
            
            loss = absolute_loss + contrastive_loss
            loss.backward()
            optimizer.step()
            
            n = inputs.size(0)
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)
            
            # --- THÊM VÀO: Tăng biến đếm toàn cục ---
            global_step += 1
            # ------------------------------------

            if i % echo_batches == echo_batches-1:
                print('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f' % (epoch + 1, i + 1, lr,  loss_absolute.avg, loss_contra.avg))
                
                # --- THÊM VÀO: Ghi lại loss của batch và hình ảnh lên TensorBoard ---
                writer.add_scalar('Train/Batch_Absolute_Loss', loss_absolute.val, global_step)
                writer.add_scalar('Train/Batch_Contrastive_Loss', loss_contra.val, global_step)
                
                # Trực quan hóa hình ảnh
                # LƯU Ý: Unsqueeze được dùng để thêm chiều kênh (C) cho ảnh xám, theo yêu cầu của add_images.
                writer.add_images('Train/Input_Images', inputs, global_step)
                writer.add_images('Train/Ground_Truth_Masks', binary_mask.unsqueeze(1), global_step)
                writer.add_images('Train/Predicted_Maps', map_x.unsqueeze(1), global_step)
                # --------------------------------------------------------

        # Ghi log loss trung bình của cả epoch
        print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.write('epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        log_file.flush()
        
        # --- THÊM VÀO: Ghi lại loss trung bình của epoch lên TensorBoard ---
        writer.add_scalar('Train/Epoch_Absolute_Loss', loss_absolute.avg, epoch)
        writer.add_scalar('Train/Epoch_Contrastive_Loss', loss_contra.avg, epoch)
        # ----------------------------------------------------
           
        epoch_test = 1
        if epoch > 25 and epoch % epoch_test == epoch_test-1:    
            model.eval()
            
            with torch.no_grad():
                ###########################################
                '''                val             '''
                ###########################################
                val_data = Spoofing_valtest(val_list, image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
                
                map_score_list = []
                
                for i, sample_batched in enumerate(dataloader_val):
                    inputs = sample_batched['image_x'].cpu()
                    string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cpu()
        
                    optimizer.zero_grad()
                    
                    map_score = 0.0
                    for frame_t in range(inputs.shape[1]):
                        map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:])
                        score_norm = torch.sum(map_x)/torch.sum(binary_mask[:,frame_t,:,:])
                        map_score += score_norm
                    map_score = map_score/inputs.shape[1]
                    
                    if map_score > 1:
                        map_score = 1.0
    
                    map_score_list.append('{} {}\n'.format( string_name[0], map_score ))
                    
                map_score_val_filename = args.log+'/'+ args.log+ '_map_score_val_%d.txt'% (epoch + 1)
                with open(map_score_val_filename, 'w') as file:
                    file.writelines(map_score_list)                
                
                torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pkl' % (epoch + 1))
        
    print('Finished Training')
    # --- THÊM VÀO: Đóng writer ---
    writer.close()
    # -------------------------------
    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.00008, help='initial learning rate')
    parser.add_argument('--batchsize', type=int, default=9, help='initial batchsize')
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')
    parser.add_argument('--epochs', type=int, default=60, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCNpp_BinaryMask_P1_07", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')

    args = parser.parse_args()
    train_test()