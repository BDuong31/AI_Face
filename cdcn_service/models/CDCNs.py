import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np


class Conv2d_cd(nn.Module):
    # Lớp Conv2d tùy chỉnh với Central Difference Convolution (CDC)
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        # Khởi tạo lớp Conv2d_cd
        super(Conv2d_cd, self).__init__() 

        # Lớp tích chập 2D tiêu chuẩn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        # Tham số theta để điều chỉnh ảnh hưởng của sự khác biệt trung tâm
        self.theta = theta

    def forward(self, x):
        # Hàm tính toán đầu ra của lớp
        out_normal = self.conv(x) # Tính toán tích chập thông thường

        # Nếu theta gần bằng 0, chỉ trả về kết quả tích chập thông thường
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace() # Điểm dừng để gỡ lỗi
            
            # Lấy kích thước của trọng số bộ lọc
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            
            # Tính tổng các trọng số theo chiều cao và chiều rộng của kernel (để tạo ra kernel_diff)
            kernel_diff = self.conv.weight.sum(2).sum(2)
            
            # Mở rộng kernel_diff để có cùng số chiều với đầu vào (thêm 2 chiều cuối)
            kernel_diff = kernel_diff[:, :, None, None]
            
            # Thực hiện tích chập với kernel_diff (đại diện cho phần central difference)
            # padding=0 vì kernel_diff hoạt động như một phép trừ điểm trung tâm
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, dilation=self.conv.dilation, groups=self.conv.groups)
            
            # Kết hợp kết quả tích chập thông thường và phần central difference
            return out_normal - self.theta * out_diff




 
        
class SpatialAttention(nn.Module):
    # Lớp Attention không gian
    def __init__(self, kernel = 3):
        # Khởi tạo lớp SpatialAttention
        super(SpatialAttention, self).__init__()

        # Lớp tích chập 2D để xử lý concatenated feature maps (avg_pool và max_pool)
        # Đầu vào là 2 kênh (từ avg_out và max_out), đầu ra là 1 kênh
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, bias=False)

        # Hàm Sigmoid để chuẩn hóa bản đồ chú ý trong khoảng (0, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Hàm tính toán đầu ra của lớp
        # Tính trung bình các giá trị theo chiều kênh (dimension 1)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        # Lấy giá trị lớn nhất theo chiều kênh (dimension 1)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Nối (concatenate) kết quả avg_out và max_out theo chiều kênh
        x = torch.cat([avg_out, max_out], dim=1)

        # Áp dụng tích chập
        x = self.conv1(x)
        
        # Áp dụng hàm sigmoid để tạo ra bản đồ chú ý không gian
        return self.sigmoid(x)



		

class CDCNpp(nn.Module):
    # Lớp CDCNpp (Central Difference Convolutional Network)
    def __init__(self, basic_conv=Conv2d_cd, theta=0.7 ):  
        # Khởi tạo lớp CDCNpp 
        super(CDCNpp, self).__init__()
        
        # Lớp tích chập đầu tiên của mạng
        self.conv1 = nn.Sequential(
            basic_conv(3, 80, kernel_size=3, stride=1, padding=1, bias=False, theta= theta), # 3 kênh đầu vào (RGB), 80 kênh đầu ra
            nn.BatchNorm2d(80), # Chuẩn hóa theo kênh
            nn.ReLU(), # Hàm kích hoạt ReLU
            
        )
        
        # Các khối mạng chính của CDCNpp
        # Khối 1
        self.Block1 = nn.Sequential(
            basic_conv(80, 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),  
            
            basic_conv(160, int(160*1.6), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.6)),
            nn.ReLU(),  
            basic_conv(int(160*1.6), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(), 
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # Giảm kích thước đầu vào xuống một nửa theo chiều cao và chiều rộng
            
        )
        
        # Khối 2
        self.Block2 = nn.Sequential(
            basic_conv(160, int(160*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),  
            basic_conv(int(160*1.2), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),  
            basic_conv(160, int(160*1.4), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.4)),
            nn.ReLU(),  
            basic_conv(int(160*1.4), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),  
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Khối 3
        self.Block3 = nn.Sequential(
            basic_conv(160, 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(), 
            basic_conv(160, int(160*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),  
            basic_conv(int(160*1.2), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Lớp tích chập cuối cùng (Original)
        # Đầu vào là concatenation của 3 khối (160*3 kênh)
        self.lastconv1 = nn.Sequential(
            basic_conv(160*3, 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            basic_conv(160, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta), # Đầu ra 1 kênh (bản đồ độ sâu hoặc mặt nạ anti-spoofing)
            nn.ReLU(), # Thường thì lớp cuối cùng cho ra bản đồ sẽ dùng Sigmoid hoặc không dùng activation nào cả, tùy thuộc vào loss function. ReLU ở đây có thể là một lựa chọn thiết kế.
        )
        
        # Các mô-đun Spatial Attention
        self.sa1 = SpatialAttention(kernel = 7)
        self.sa2 = SpatialAttention(kernel = 5)
        self.sa3 = SpatialAttention(kernel = 3)

        # Lớp downsample để giảm kích thước đầu ra của các khối xuống 32x32
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x # Lưu trữ đầu vào để sử dụng sau này (có thể là để so sánh hoặc phân tích)
        x = self.conv1(x) # Thực hiện tích chập đầu tiên
        
        # Xử lý qua Khối 1 và Spatial Attention
        x_Block1 = self.Block1(x)	    	    	
        attention1 = self.sa1(x_Block1) # Tính toán bản đồ chú ý không gian cho khối 1
        x_Block1_SA = attention1 * x_Block1 # Nhân theo từng phần tử (element-wise) feature map với bản đồ chú ý
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)  # Giảm/tăng kích thước của x_Block1_SA về 32x32
        
        # Xử lý qua Khối 2 và Spatial Attention
        x_Block2 = self.Block2(x_Block1)  # Đầu vào của Block2 là đầu ra của Block1 (không qua SA)	    
        attention2 = self.sa2(x_Block2)  
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)  
        
        # Xử lý qua Khối 3 và Spatial Attention
        x_Block3 = self.Block3(x_Block2) # Đầu vào của Block3 là đầu ra của Block2 (không qua SA)  
        attention3 = self.sa3(x_Block3)  
        x_Block3_SA = attention3 * x_Block3	
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)   
        
        # Nối các đầu ra của các khối lại với nhau
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), dim=1)    
        
        #pdb.set_trace() # Điểm dừng để gỡ lỗi
        
        # Thực hiện tích chập cuối cùng để tạo ra bản đồ độ sâu hoặc mặt nạ anti-spoofing
        map_x = self.lastconv1(x_concat)
        
        # Loại bỏ chiều kênh (dimension 1) nếu nó có kích thước là 1
        map_x = map_x.squeeze(1)
        
        # Trả về bản đồ kết quả, feature map đã nối, các bản đồ chú ý và đầu vào ban đầu
        return map_x, x_concat, attention1, attention2, attention3, x_input

