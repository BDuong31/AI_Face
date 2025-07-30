from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa

# tăng dữ liệu từ 'imgaug' --> thêm (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Thêm màu
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast với gamma từ 0.5 đến 1.5
])

# mảng numpy
class RandomErasing(object):
    '''
    Lớp thực hiện Random Erasing (Xóa ngẫu nhiên) trong tăng cường dữ liệu
    -------------------------------------------------------------------------------------
    probability: Xác suất thực hiện thao tác này.
    sl: vùng xóa tối thiểu
    sh: vùng xóa tối đa
    r1:tỷ lệ khung hình tối thiểu
    mean: giá trị xóa
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability # Xác suất thực hiện thao tác xóa ngẫu nhiên
        self.mean = mean # Giá trị trung bình để thay thế vùng xóa
        self.sl = sl # Vùng xóa tối thiểu (tính theo tỷ lệ so với diện tích ảnh)
        self.sh = sh # Vùng xóa tối đa (tính theo tỷ lệ so với diện tích ảnh)
        self.r1 = r1 # Tỷ lệ khung hình tối thiểu (tỷ lệ chiều cao/chiều rộng của vùng xóa)
       
    def __call__(self, sample):
        # Hàm gọi khi áp dụng lớp này lên một mẫu dữ liệu
        img, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        # Kiểm tra xem xác suất có đủ để thực hiện thao tác xóa ngẫu nhiên
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3) # Số lần thử để tìm vị trí xóa hợp lệ
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1] # Tính diện tích của ảnh
           
                # Chọn ngẫu nhiên diện tích mục tiêu cho vùng xóa
                target_area = random.uniform(self.sl, self.sh) * area

                # Chọn ngẫu nhiên tỷ lệ khung hình cho vùng xóa
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                # Tính chiều cao và chiều rộng của vùng xóa dựa trên diện tích mục tiêu và tỷ lệ khung hình
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                # Kiểm tra xem vùng xóa có nằm trong kích thước ảnh không
                if w < img.shape[1] and h < img.shape[0]:

                    # Chọn ngẫu nhiên vị trí bắt đầu của vùng xóa
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    # Tạo mặt nạ nhị phân cho vùng xóa
                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
        # Trả về mẫu đã được xử lý
        return {'image_x': img, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    '''
        Lớp thực hiện Cutout: xóa một vùng hình chữ nhật ngẫu nhiên trên ảnh (tensor).
        length: Độ dài tối đa của cạnh hình chữ nhật xóa.
    '''
    def __init__(self, length=50):
        self.length = length   # Độ dài tối đa của cạnh hình chữ nhật xóa

    def __call__(self, sample):
        img, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        # Lấy chiều cao (h) và chiều rộng (w) của ảnh tensor
        # Đối với tensor PyTorch (C x H x W), img.shape[1] là H, img.shape[2] là W
        # Đối với mảng numpy (H x W x C), img.shape[0] là H, img.shape[1] là W
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)  # Tạo mặt nạ với kích thước (h, w) và giá trị 1.0
        
        # Chọn ngẫu nhiên vị trí (y, x) để bắt đầu xóa
        y = np.random.randint(h)
        x = np.random.randint(w)

        # Chọn độ dài mới ngẫu nhiên trong khoảng từ 1 đến self.length
        length_new = np.random.randint(1, self.length)
        
        # Tính toán các tọa độ của hình chữ nhật xóa
        # Sử dụng np.clip để đảm bảo các tọa độ không vượt quá kích thước ảnh
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0. # Đặt giá trị 0 cho vùng xóa trong mặt nạ
        mask = torch.from_numpy(mask) # Chuyển đổi mặt nạ sang tensor PyTorch
        mask = mask.expand_as(img) # Mở rộng mặt nạ để có cùng kích thước với ảnh tensor
        img *= mask # Nhân ảnh tensor với mặt nạ để xóa vùng hình chữ nhật
        return {'image_x': img, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        Chuẩn hóa ảnh về khoảng [-1, 1], tương tự như trong MXNet.
        Công thức: image = (image - 127.5) / 128
        Giả định giá trị pixel ban đầu của ảnh nằm trong khoảng [0, 255].
    """
    def __call__(self, sample):
        image_x, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        new_image_x = (image_x - 127.5)/128  # # Chuẩn hóa ảnh về khoảng [-1, 1]

        return {'image_x': new_image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}



class RandomHorizontalFlip(object):
    """Lật ngang ảnh (Image) một cách ngẫu nhiên với xác suất 0.5."""
    def __call__(self, sample):
        image_x, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        # Khởi tạo ảnh và mặt nạ mới (có thể không cần thiết nếu không lật)
        new_image_x = np.zeros((256, 256, 3)) 
        new_binary_mask = np.zeros((32, 32))

        p = random.random() # Tạo một số ngẫu nhiên trong khoảng [0, 1]
        if p < 0.5: # Nếu số ngẫu nhiên nhỏ hơn 0.5
            #print('Flip') 

            new_image_x = cv2.flip(image_x, 1)  # Lật ngang ảnh (1 là lật ngang, 0 là lật dọc)
            new_binary_mask = cv2.flip(binary_mask, 1) # Lật ngang mặt nạ nhị phân
           
                
            return {'image_x': new_image_x, 'binary_mask': new_binary_mask, 'spoofing_label': spoofing_label}
        else: # không lật ảnh
            #print('no Flip')
            return {'image_x': image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}



class ToTensor(object):
    """
        Chuyển đổi các mảng ndarray trong sample thành Tensor PyTorch.
        Xử lý một batch mỗi lần (thực ra là một sample mỗi lần trong ngữ cảnh Dataset).
    """

    def __call__(self, sample):
        image_x, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        # Đảo ngược thứ tự kênh màu và hoán vị các chiều của ảnh
        # numpy image: (batch_size) x H x W x C (chiều cao x chiều rộng x kênh)
        # torch image: (batch_size) x C X H X W (kênh x chiều cao x chiều rộng)
        # image_x[:,:,::-1] đổi từ BGR (OpenCV mặc định) sang RGB
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x) # Đảm bảo ảnh là mảng numpy
        
        binary_mask = np.array(binary_mask) # Đảm bảo mặt nạ nhị phân là mảng numpy

        # Tạo một mảng numpy cho nhãn spoofing, sau đó chuyển thành tensor
        # Chuyển đổi nhãn spoofing thành mảng numpy với dtype int64
        spoofing_label_np = np.array([0],dtype=np.int64)
        spoofing_label_np[0] = spoofing_label
        
        # Chuyển đổi các mảng numpy thành tensor PyTorch
        return {'image_x': torch.from_numpy(image_x.astype(np.float64)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(np.float64)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.float64)).float()}


# --- Lớp Dataset cho dữ liệu huấn luyện Spoofing ---
class Spoofing_train(Dataset):
    """
    Lớp Dataset tùy chỉnh cho việc huấn luyện mô hình phát hiện spoofing.
    Đọc thông tin từ một tệp CSV và tải ảnh tương ứng.
    """
    def __init__(self, info_list, root_dir,  transform=None):
        """
        Hàm khởi tạo.
        info_list: Đường dẫn đến tệp CSV chứa thông tin các mẫu (tên video/ảnh, nhãn).
        root_dir: Thư mục gốc chứa các tệp ảnh/video.
        transform: Các phép biến đổi (augmentation) sẽ được áp dụng cho mỗi mẫu.
        """
        # Đọc tệp CSV chứa thông tin về các mẫu
        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir # Thư mục gốc chứa các tệp ảnh/video
        self.transform = transform # Các phép biến đổi sẽ được áp dụng cho mỗi mẫu

    def __len__(self):
        # Trả về số lượng mẫu trong tập dữ liệu
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu tại chỉ số idx.
        """

        #print(self.landmarks_frame.iloc[idx, 0]) # lấy tên video/ảnh từ tệp CSV
        
        # Lấy tên video/ảnh từ tệp CSV và tạo đường dẫn đầy đủ đến tệp ảnh
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)
    
        # Lấy ảnh và mặt nạ nhị phân từ hàm get_single_image_x
        image_x, binary_mask = self.get_single_image_x(image_path)
        
        # Lấy nhãn spoofing từ tệp CSV
        spoofing_label = self.landmarks_frame.iloc[idx, 1]
        if spoofing_label == 1:
            spoofing_label = 1            # real (thật)
        else:
            spoofing_label = 0            # fake (giả)
            binary_mask = np.zeros((32, 32))  # Nếu là giả, đặt mặt nạ nhị phân thành 0 (không có mặt nạ)
        
        # Lấy nhãn spoofing từ tệp CSV
        #frequency_label = self.landmarks_frame.iloc[idx, 2:2+50].values  

        # Tạo một mẫu dữ liệu với ảnh, mặt nạ nhị phân và nhãn spoofing
        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}

        # Nếu có phép biến đổi (augmentation), áp dụng chúng lên mẫu
        if self.transform:
            sample = self.transform(sample)
        return sample # Trả về mẫu đã được xử lý

    def get_single_image_x(self, image_path):
        """
            Đọc một ảnh từ đường dẫn, thay đổi kích thước, tăng cường và tạo binary mask.
        """
        
        # Khởi tạo ảnh và binary mask với giá trị 0
        image_x = np.zeros((256, 256, 3)) # Ảnh màu kích thước 256x256 với 3 kênh màu
        binary_mask = np.zeros((32, 32)) # Mặt nạ nhị phân kích thước 32x32
 
 
        # Đọc ảnh từ đường dẫn
        image_x_temp = cv2.imread(image_path)

        # Đọc ảnh ở chế độ xám (gray) để tạo mặt nạ nhị phân
        image_x_temp_gray = cv2.imread(image_path, 0)

        image_x = cv2.resize(image_x_temp, (256, 256)) # Thay đổi kích thước ảnh về 256x256
        image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32)) # Thay đổi kích thước mặt nạ nhị phân về 32x32
        image_x_aug = seq.augment_image(image_x)  # Tăng cường ảnh bằng imgaug
            
        # Tạo mặt nạ nhị phân từ ảnh xám
        # Nếu giá trị pixel lớn hơn 0, đặt giá trị tương ứng trong mặt nạ nhị phân là 1, ngược lại là 0
        for i in range(32):
            for j in range(32):
                if image_x_temp_gray[i,j]>0:
                    binary_mask[i,j]=1
                else:
                    binary_mask[i,j]=0
        
        # Trả về ảnh đã tăng cường và binary mask
        return image_x_aug, binary_mask