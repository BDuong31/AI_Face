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


frames_total = 8   # mỗi video sẽ lấy 8 mẫu frame một cách đồng đều (uniform samples)


class Normaliztion_valtest(object):
    """
        Chuẩn hóa ảnh tương tự như trong MXNet, đưa giá trị pixel về khoảng [-1, 1].
        Công thức: image = (image - 127.5) / 128
        Được sử dụng cho tập validation và test.
    """
    def __call__(self, sample):
        # sample là một từ điển chứa các thành phần của ảnh và nhãn
        image_x, binary_mask, string_name = sample['image_x'],sample['binary_mask'],sample['string_name']
        
        # Chuẩn hóa giá trị pixel của ảnh về khoảng [-1,1]
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        
        return {'image_x': new_image_x, 'binary_mask': binary_mask, 'string_name': string_name}


class ToTensor_valtest(object):
    """
        Chuyển đổi các mảng ndarray trong sample thành Tensor PyTorch.
        Được sử dụng cho tập validation và test.
        Xử lý một batch mỗi lần (thực ra là một sample mỗi lần trong ngữ cảnh Dataset).
    """

    def __call__(self, sample):
        image_x, binary_mask, string_name = sample['image_x'],sample['binary_mask'],sample['string_name']
        
        # Đảo ngược thứ tự kênh màu (BGR sang RGB) và hoán vị các chiều của ảnh
        # Giả định image_x có dạng: (Số frames (T) x Chiều cao (H) x Chiều rộng (W) x Kênh (C))
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        # image_x[:,:,:,::-1] đổi từ BGR (OpenCV mặc định) sang RGB cho mỗi frame
        # .transpose((0, 3, 1, 2)) thay đổi từ (T, H, W, C) sang (T, C, H, W)
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x) # Đảm bảo image_x là numpy array
                        
        binary_mask = np.array(binary_mask) # Đảm bảo binary_mask là numpy array
        
        # Chuyển đổi các mảng numpy thành Tensor PyTorch
        return {'image_x': torch.from_numpy(image_x.astype(np.float64)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(np.float64)).float(), 'string_name': string_name} 



class Spoofing_valtest(Dataset):
    """
    Lớp Dataset tùy chỉnh cho việc validation và test mô hình phát hiện spoofing.
    Đọc thông tin từ một tệp CSV, tải một số lượng frame cố định từ mỗi video.
    """
    def __init__(self, info_list, root_dir,  transform=None):
        """
        Hàm khởi tạo.
        info_list: Đường dẫn đến tệp CSV chứa thông tin các video (tên video).
        root_dir: Thư mục gốc chứa các thư mục video, mỗi thư mục video chứa các frame.
        transform: Các phép biến đổi (ví dụ: chuẩn hóa, chuyển sang tensor) sẽ được áp dụng cho mỗi mẫu.
        """

        # Đọc tệp CSV chứa thông tin video
        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir # Thư mục gốc chứa các video
        self.transform = transform # Các phép biến đổi sẽ được áp dụng cho mỗi mẫu

    def __len__(self):

        # Trả về số lượng video trong tập dữ liệu
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu (một video với các frame đã chọn) tại chỉ số idx.
        """

        # Lấy tên video từ tệp CSV
        videoname = str(self.landmarks_frame.iloc[idx, 0])

        # Tạo đường dẫn đến thư mục chứa các frame của video
        image_path = os.path.join(self.root_dir, videoname)

        # Kiểm tra xem đường dẫn có kết thúc bằng dấu '/' không, nếu không thì thêm vào
        image_path = os.path.join(image_path, '')
             
        # Lấy các frame ảnh và binary mask tương ứng từ thư mục video
        image_x, binary_mask = self.get_single_image_x(image_path, videoname)
		    
        # Tạo một dictionary chứa dữ liệu của mẫu
        # 'string_name' được dùng để lưu tên video, hữu ích cho việc theo dõi hoặc gỡ lỗi
        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'string_name': videoname}

        # Nếu có phép biến đổi nào được định nghĩa, áp dụng chúng lên mẫu
        if self.transform:
            sample = self.transform(sample)
        return sample # trả về mẫu đã được biến đổi

    def get_single_image_x(self, image_path, videoname):
        """
        Đọc 'frames_total' (ví dụ 8) frame từ thư mục video được chỉ định.
        Các frame được chọn một cách đồng đều.
        Tạo binary mask cho mỗi frame.
        image_path: Đường dẫn đến thư mục chứa các frame của một video.
        videoname: Tên video (có thể dùng để gỡ lỗi hoặc trong trường hợp tên tệp frame phức tạp).
        """

        # Đếm tổng số tệp (frame) trong thư mục video
        # Chỉ đếm các tệp, không đếm thư mục con
        files_total = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
        
        interval = files_total//frames_total # lấy khoảng cách giữa các frame để đảm bảo chọn được 'frames_total' frame một cách đồng đều
         
        image_x = np.zeros((frames_total, 256, 256, 3)) # RGB ảnh sẽ có kích thước 256x256
        
        binary_mask = np.zeros((frames_total, 32, 32)) # khởi tạo binary mask với kích thước 32x32 cho mỗi frame
        
        
        
        # ngẫu nhiên chọn các frame từ video
        for ii in range(frames_total):

            # Tính chỉ số của frame cần lấy
            image_id = ii*interval + 1 
            
            # Đảm bảo chỉ số frame nằm trong khoảng hợp lệ
            s = "%04d.jpg" % image_id            
            
            # RGB
            image_path2 = os.path.join(image_path, s)
            image_x_temp = cv2.imread(image_path2)
            
            image_x_temp_gray = cv2.imread(image_path2, 0) # đọc ảnh ở chế độ xám
            image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32)) # resize về kích thước 32x32


            image_x[ii,:,:,:] = cv2.resize(image_x_temp, (256, 256)) # resize về kích thước 256x256
            
            #print(image_path2)
            
            # Tạo binary mask từ ảnh xám
            for i in range(32):
                for j in range(32):
                    if image_x_temp_gray[i,j]>0:
                        binary_mask[ii, i, j]=1.0
                    else:
                        binary_mask[ii, i, j]=0.0
            

        
        return image_x, binary_mask







if __name__ == '__main__':
    # Ví dụ sử dụng (Lưu ý: Các lớp và đường dẫn này có thể dành cho một tập dữ liệu khác tên là BioVid)
    # MAHNOB (Tên này có thể là một tập dữ liệu khác hoặc một phần của dự án)
    root_list = '/wrk/yuzitong/DONOTREMOVE/BioVid_Pain/data/cropped_frm/' # Đường dẫn đến thư mục chứa các frame đã crop
    trainval_list = '/wrk/yuzitong/DONOTREMOVE/BioVid_Pain/data/ImageSet_5fold/trainval_zitong_fold1.txt' # Đường dẫn đến tệp chứa danh sách các video trong tập trainval
    
    # Tạo đối tượng BioVid_train từ lớp Spoofing_valtest
    BioVid_train = BioVid(trainval_list, root_list, transform=transforms.Compose([Normaliztion(), Rescale((133,108)),RandomCrop((125,100)),RandomHorizontalFlip(),  ToTensor()]))
    
    # Tạo DataLoader để tải dữ liệu theo lô (batch)
    dataloader = DataLoader(BioVid_train, batch_size=1, shuffle=True, num_workers=8)
    
    # Duyệt qua các lô dữ liệu trong DataLoader
    for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched['image_x'].size(), sample_batched['video_label'].size())
        print(i_batch, sample_batched['image_x'], sample_batched['pain_label'], sample_batched['ecg'])
        pdb.set_trace()
        break

            
 


