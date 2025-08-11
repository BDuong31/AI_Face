import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from models.CDCNs import Conv2d_cd, CDCNpp
import argparse
import random
from PIL import Image
import glob

# Thiết lập để sử dụng CPU
device = torch.device('cpu')

def preprocess_image(image_path, size=256):
    """Tiền xử lý ảnh đầu vào để phù hợp với mô hình"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    
    # Chuyển đổi từ BGR sang RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Nếu tìm thấy khuôn mặt, cắt ảnh
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Lấy khuôn mặt đầu tiên
        face_img = image[y:y+h, x:x+w]
    else:
        face_img = image  # Nếu không tìm thấy, sử dụng ảnh gốc
    
    # Resize ảnh
    face_img = cv2.resize(face_img, (size, size))
    
    # Chuyển đổi thành tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Chuyển sang PIL Image và áp dụng transform
    pil_img = Image.fromarray(face_img)
    tensor_img = transform(pil_img).unsqueeze(0)  # Thêm batch dimension
    
    return tensor_img, face_img

def load_model(model_path='CDCNpp_BinaryMask_P1_07/CDCNpp_BinaryMask_P1_07_30.pkl', theta=0.7):
    """Tải mô hình CDCN++ đã huấn luyện"""
    model = CDCNpp(basic_conv=Conv2d_cd, theta=theta)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_one_image(model, image_path, threshold=0.5):
    """Dự đoán một ảnh và trả về kết quả"""
    try:
        # Tiền xử lý ảnh
        img_tensor, original_img = preprocess_image(image_path)
        img_tensor = img_tensor.to(device)
        
        # Dự đoán
        with torch.no_grad():
            map_x, embedding, attention1, attention2, attention3, x_input = model(img_tensor)
            
            # Tính điểm và dự đoán
            score = torch.mean(map_x).item()
            prediction = "Thật" if score <= threshold else "Giả mạo"
            
            # Chuẩn hóa map_x để hiển thị
            depth_map = map_x.cpu().numpy()[0]
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            
            # Chuẩn hóa các attention maps
            att1 = attention1.cpu().numpy()[0, 0]
            att2 = attention2.cpu().numpy()[0, 0]
            att3 = attention3.cpu().numpy()[0, 0]
            
            return {
                'image': original_img,
                'score': score,
                'prediction': prediction,
                'depth_map': depth_map,
                'attention1': att1,
                'attention2': att2,
                'attention3': att3
            }
    except Exception as e:
        print(f"Lỗi khi dự đoán ảnh {image_path}: {e}")
        return None

def visualize_prediction(result, image_path, save_path=None):
    """Trực quan hóa kết quả dự đoán"""
    if result is None:
        return
    
    # Thiết lập kích thước hình
    plt.figure(figsize=(16, 8))
    
    # Hiển thị ảnh gốc
    plt.subplot(2, 3, 1)
    plt.imshow(result['image'])
    plt.title(f"Ảnh đầu vào\n{os.path.basename(image_path)}", fontsize=12)
    plt.axis('off')
    
    # Hiển thị kết quả dự đoán
    plt.subplot(2, 3, 2)
    plt.imshow(result['image'])
    plt.title(f"Dự đoán: {result['prediction']}\nĐiểm: {result['score']:.4f}", fontsize=12)
    color = 'green' if result['prediction'] == "Thật" else 'red'
    plt.text(10, 30, result['prediction'], color=color, fontsize=20, weight='bold')
    plt.axis('off')
    
    # Hiển thị depth map
    plt.subplot(2, 3, 3)
    plt.imshow(result['depth_map'], cmap='jet')
    plt.title("Bản đồ độ sâu", fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Hiển thị attention maps
    plt.subplot(2, 3, 4)
    plt.imshow(result['attention1'], cmap='viridis')
    plt.title("Attention map 1", fontsize=12)
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(result['attention2'], cmap='viridis')
    plt.title("Attention map 2", fontsize=12)
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(result['attention3'], cmap='viridis')
    plt.title("Attention map 3", fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Đã lưu kết quả vào {save_path}")
    else:
        plt.show()
    
    plt.close()

def process_test_images(model, image_dir, output_dir, threshold=0.5, num_samples=5):
    """Xử lý nhiều ảnh test và lưu kết quả"""
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm tất cả ảnh trong thư mục
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_paths:
        print(f"Không tìm thấy ảnh trong thư mục {image_dir}")
        return
    
    # Lấy ngẫu nhiên một số lượng ảnh
    if len(image_paths) > num_samples:
        image_paths = random.sample(image_paths, num_samples)
    
    # Xử lý từng ảnh
    for i, image_path in enumerate(image_paths):
        print(f"Đang xử lý ảnh {i+1}/{len(image_paths)}: {image_path}")
        
        result = predict_one_image(model, image_path, threshold)
        if result:
            save_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}.png")
            visualize_prediction(result, image_path, save_path)
    
    print(f"\nĐã xử lý {len(image_paths)} ảnh. Kết quả được lưu trong thư mục {output_dir}")

def visualize_model_architecture():
    """Trực quan hóa kiến trúc mô hình CDCN++"""
    model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
    
    # Tạo biểu đồ kiến trúc đơn giản
    plt.figure(figsize=(12, 10))
    
    # Định nghĩa các khối và kích thước
    blocks = [
        {"name": "Input Image", "size": "3x256x256", "color": "lightgray"},
        {"name": "Conv1", "size": "80x256x256", "color": "lightblue"},
        {"name": "Block1 + SA", "size": "160x128x128", "color": "lightgreen"},
        {"name": "Block2 + SA", "size": "160x64x64", "color": "lightcoral"},
        {"name": "Block3 + SA", "size": "160x32x32", "color": "lightyellow"},
        {"name": "Concat + Downsample", "size": "480x32x32", "color": "lightpink"},
        {"name": "LastConv", "size": "1x32x32", "color": "skyblue"},
        {"name": "Output Depth Map", "size": "32x32", "color": "lightgray"}
    ]
    
    # Vẽ các khối
    box_height = 0.8
    for i, block in enumerate(blocks):
        y = 9 - i  # Vị trí từ trên xuống dưới
        
        # Vẽ khối
        plt.fill_between([2, 10], [y-box_height/2, y-box_height/2], [y+box_height/2, y+box_height/2], color=block["color"])
        
        # Thêm tên và kích thước
        plt.text(6, y, f"{block['name']}\n{block['size']}", ha='center', va='center', fontsize=12)
        
        # Thêm mũi tên kết nối
        if i < len(blocks) - 1:
            plt.arrow(6, y-box_height/2, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Thêm tiêu đề
    plt.text(6, 10, "Kiến trúc mô hình CDCN++", ha='center', va='center', fontsize=16, weight='bold')
    
    # Thêm chú thích bên phải
    notes = [
        "Central Difference Convolution (CDC):",
        "- Kết hợp tích chập thông thường và",
        "  chênh lệch trung tâm",
        "- Tham số theta=0.7 điều chỉnh mức độ ảnh hưởng",
        "",
        "Spatial Attention (SA):",
        "- Tăng cường vùng quan trọng",
        "- Kết hợp max pooling và avg pooling",
        "",
        "Multi-level Feature Fusion:",
        "- Kết hợp đặc trưng từ nhiều tầng",
        "- Giúp phát hiện giả mạo ở nhiều mức độ chi tiết"
    ]
    
    for i, note in enumerate(notes):
        plt.text(12, 9-i*0.5, note, fontsize=10, ha='left', va='center')
    
    plt.xlim(0, 18)
    plt.ylim(0, 11)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('cdcn_architecture.png', dpi=200)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa mô hình CDCN++ và dự đoán ảnh")
    parser.add_argument('--mode', type=str, default='all', choices=['metrics', 'test', 'architecture', 'all'],
                        help='Chế độ hoạt động: metrics (vẽ biểu đồ metrics), test (kiểm tra ảnh), architecture (trực quan hóa kiến trúc), all (tất cả)')
    parser.add_argument('--model', type=str, default='CDCNpp_BinaryMask_P1_07/CDCNpp_BinaryMask_P1_07_30.pkl',
                        help='Đường dẫn đến file mô hình đã huấn luyện')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Ngưỡng để phân loại thật/giả')
    parser.add_argument('--image_dir', type=str, default='datamain/test',
                        help='Thư mục chứa ảnh test')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                        help='Thư mục để lưu kết quả trực quan hóa')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Số lượng ảnh mẫu để kiểm tra')
    parser.add_argument('--theta', type=float, default=0.7,
                        help='Tham số theta cho mô hình CDCN++')
    
    args = parser.parse_args()
    
    # Tạo thư mục đầu ra
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ['metrics', 'all']:
        try:
            from visualize_metrics import main as visualize_metrics_main
            visualize_metrics_main()
        except ImportError:
            print("Không tìm thấy module visualize_metrics. Hãy chạy script visualize_metrics.py riêng.")
    
    if args.mode in ['architecture', 'all']:
        print("Đang tạo biểu đồ kiến trúc mô hình...")
        visualize_model_architecture()
    
    if args.mode in ['test', 'all']:
        print("Đang tải mô hình...")
        model = load_model(args.model, args.theta)
        
        print("Đang kiểm tra mô hình trên ảnh test...")
        process_test_images(model, args.image_dir, args.output_dir, args.threshold, args.num_samples)

if __name__ == "__main__":
    main()
