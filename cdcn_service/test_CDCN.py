from __future__ import print_function, division
import torch
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from models.CDCNs import Conv2d_cd, CDCNpp

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
test_list = os.path.join(image_dir, 'protocol', 'test_list.txt')
# image_dir = '/Users/bduong/Documents/CDCN/FAS_challenge_CVPRW2020/Track2_Single-modal/model1_pytorch/my_face_dataset'  
   

# test_list =  '/Users/bduong/Documents/CDCN/FAS_challenge_CVPRW2020/Track2_Single-modal/model1_pytorch/my_face_dataset/protocol/test_list.txt'

#test_list =  '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@2_test_res.txt'
#test_list =  '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@3_test_res.txt'


# main function
def train_test():


    print("Đánh giá mô hình CDCNpp:\n ")

     
    #model = CDCNpp( basic_conv=Conv2d_cd, theta=0.7)

    # Khởi tạo mô hình CDCNpp với Conv2d_cd và tham số theta từ đối số
    model = CDCNpp( basic_conv=Conv2d_cd, theta=args.theta)
    model.load_state_dict(torch.load('./CDCNpp_BinaryMask_P1_07/CDCNpp_BinaryMask_P1_07_30.pkl'))


    # Chuyển mô hình sang chế độ đánh giá (eval) và chuyển sang CPU
    model = model.cpu()

    print(model) 

    model.eval()
    
    # Không cần tính toán gradient trong quá trình đánh giá
    with torch.no_grad(): 
        ##########################################################################
        '''                Đánh giá trên tập test và tính toán map score       '''
        ##########################################################################
        # Ngưỡng để xác định kết quả là thật hay giả
        val_data = Spoofing_valtest(test_list, image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
        
        # Khởi tạo danh sách để lưu trữ điểm số bản đồ (map score)
        map_score_list = []
        
        # Duyệt qua từng lô dữ liệu trong DataLoader
        for i, sample_batched in enumerate(dataloader_val):
            
            print(i)
            
            # Lấy đầu vào (inputs) và các thông tin khác từ lô dữ liệu
            inputs = sample_batched['image_x'].cpu() # 
            string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cpu()

            map_score = 0.0

            # Duyệt qua từng frame trong đầu vào
            for frame_t in range(inputs.shape[1]):

                # Lấy dữ liệu của frame hiện tại: inputs[:,frame_t,:,:,:]
                # Đưa frame qua mô hình để nhận đầu ra
                # map_x: bản đồ dự đoán (prediction map)
                # embedding, x_Block1, x_Block2, x_Block3, x_input: các feature map trung gian (nếu cần)
                map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:])

                # Tính điểm score cho frame hiện tại bằng cách chuẩn hóa tổng giá trị của map_x
                # chia cho tổng giá trị của binary_mask (vùng khuôn mặt)
                # Điều này giúp tập trung vào vùng quan trọng (khuôn mặt)
                score_norm = torch.sum(map_x)/torch.sum(binary_mask[:,frame_t,:,:])

                map_score += score_norm # Cộng dồn điểm số của từng frame
            map_score = map_score/inputs.shape[1] # Tính trung bình điểm số của tất cả các frame
            

            if map_score>1:
                map_score = 1.0

            print('map_score:', map_score)

            # Lưu trữ điểm số bản đồ vào danh sách
            map_score_list.append('{} {}\n'.format( string_name[0], map_score ))
        
        # Lưu danh sách điểm số bản đồ vào tệp
        map_score_val_filename = args.log+'/'+ args.log+ '_map_score_test_50.txt'
        with open(map_score_val_filename, 'w') as file:
            file.writelines(map_score_list)                
                

    print('Hoàn thành đánh giá mô hình CDCNpp trên tập dữ liệu test.')
  

  
 

if __name__ == "__main__":

    # Tạo đối tượng ArgumentParser để định nghĩa và xử lý các tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Đánh giá mô hình CDCNpp cho phát hiện giả mạo khuôn mặt")
    parser.add_argument('--gpu', type=int, default=3, help='id gpu được sử dụng để dự đoán')  
    parser.add_argument('--lr', type=float, default=0.0001, help='tốc độ học tập ban đầu')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=9, help='kích thước ảnh ban đầu')  #default=7  
    parser.add_argument('--step_size', type=int, default=20, help='có bao nhiêu kỷ nguyên lr phân rã một lần')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma của optim.lr_scheduler.StepLR, sự suy giảm của lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='có bao nhiêu ảnh hiển thị một lần')  # 50
    parser.add_argument('--epochs', type=int, default=50, help='tổng số thời gian đào tạo')
    parser.add_argument('--log', type=str, default="CDCNpp_BinaryMask_P1_07", help='ghi nhật ký và lưu tên mô hình')
    parser.add_argument('--finetune', action='store_true', default=False, help='có tinh chỉnh các mô hình khác không')
    parser.add_argument('--theta', type=float, default=0.7, help='siêu tham số trong CDCNpp')

    # Phân tích các tham số dòng lệnh
    args = parser.parse_args()
    train_test()
