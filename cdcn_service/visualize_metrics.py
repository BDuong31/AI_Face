import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from utils import get_threshold, test_threshold_based, performances
import pandas as pd
from sklearn.metrics import roc_curve, auc

# Thư mục chứa dữ liệu
log_dir = "CDCNpp_BinaryMask_P1_07"

def parse_train_log():
    """Đọc và phân tích file log huấn luyện để lấy thông tin về loss"""
    log_file_path = os.path.join(log_dir, f"{log_dir}_log.txt")
    if not os.path.exists(log_file_path):
        print(f"Không tìm thấy file log: {log_file_path}")
        return None, None
    
    # Đọc file log
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Khởi tạo danh sách để lưu dữ liệu
    epochs = []
    absolute_losses = []
    contrastive_losses = []
    
    # Phân tích các dòng log để lấy thông tin loss
    pattern = r'epoch:(\d+), Train: Absolute_Depth_loss= ([\d\.]+), Contrastive_Depth_loss= ([\d\.]+)'
    for line in lines:
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            abs_loss = float(match.group(2))
            cont_loss = float(match.group(3))
            
            epochs.append(epoch)
            absolute_losses.append(abs_loss)
            contrastive_losses.append(cont_loss)
    
    return epochs, absolute_losses, contrastive_losses

def get_val_scores():
    """Lấy điểm số đánh giá từ tất cả các epoch"""
    map_score_files = glob.glob(os.path.join(log_dir, f"{log_dir}_map_score_val_*.txt"))
    
    # Sắp xếp file theo số epoch
    def get_epoch_num(filename):
        match = re.search(r'val_(\d+)\.txt$', filename)
        if match:
            return int(match.group(1))
        return 0
    
    map_score_files = sorted(map_score_files, key=get_epoch_num)
    
    result_data = []
    epochs = []
    
    for file_path in map_score_files:
        epoch = get_epoch_num(file_path)
        if epoch == 0:
            continue  # Bỏ qua nếu không xác định được epoch
        
        epochs.append(epoch)
        
        # Đọc dữ liệu từ file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Phân tách dữ liệu: filename và score
        scores = []
        labels = []
        
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) >= 2:
                # Giả định rằng dòng cuối cùng là score và dòng trước đó chứa label
                # Nếu format khác, bạn cần điều chỉnh
                score = float(tokens[-1])
                # Giả định: 1 = thật, 0 = giả
                label = 1 if "real" in tokens[0].lower() else 0
                
                scores.append(score)
                labels.append(label)
        
        # Tính toán metrics
        if len(scores) > 0 and len(labels) > 0:
            fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
            auc_score = auc(fpr, tpr)
            
            # Tìm threshold tối ưu
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Tính toán các metric
            TP = sum((np.array(scores) >= optimal_threshold) & (np.array(labels) == 1))
            TN = sum((np.array(scores) < optimal_threshold) & (np.array(labels) == 0))
            FP = sum((np.array(scores) >= optimal_threshold) & (np.array(labels) == 0))
            FN = sum((np.array(scores) < optimal_threshold) & (np.array(labels) == 1))
            
            total_real = sum(np.array(labels) == 1)
            total_fake = sum(np.array(labels) == 0)
            
            accuracy = (TP + TN) / len(labels) if len(labels) > 0 else 0
            apcer = FP / total_fake if total_fake > 0 else 0  # Tỷ lệ giả bị phân loại sai
            bpcer = FN / total_real if total_real > 0 else 0  # Tỷ lệ thật bị phân loại sai
            acer = (apcer + bpcer) / 2  # Trung bình lỗi phân loại
            
            result_data.append({
                'epoch': epoch,
                'threshold': optimal_threshold,
                'accuracy': accuracy,
                'apcer': apcer,
                'bpcer': bpcer,
                'acer': acer,
                'auc': auc_score,
                'fpr': fpr,
                'tpr': tpr
            })
    
    return epochs, result_data

def plot_loss_curves(epochs, absolute_losses, contrastive_losses):
    """Vẽ đồ thị loss theo thời gian"""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, absolute_losses, 'b-', label='Absolute Depth Loss')
    plt.plot(epochs, contrastive_losses, 'r-', label='Contrastive Depth Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss của mô hình CDCN++ theo thời gian huấn luyện', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'loss_curves.png'), dpi=300)
    plt.show()

def plot_metrics(epochs, result_data):
    """Vẽ đồ thị các metrics theo thời gian"""
    accuracies = [d['accuracy'] for d in result_data]
    apcers = [d['apcer'] for d in result_data]
    bpcers = [d['bpcer'] for d in result_data]
    acers = [d['acer'] for d in result_data]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, accuracies, 'g-o', label='Độ chính xác (ACC)')
    plt.plot(epochs, [1-a for a in acers], 'b-^', label='Hiệu suất chống giả mạo (1-ACER)')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Tỷ lệ', fontsize=12)
    plt.title('Độ chính xác và hiệu suất chống giả mạo của mô hình CDCN++', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, apcers, 'r-s', label='APCER (Tỷ lệ giả → thật)')
    plt.plot(epochs, bpcers, 'm-d', label='BPCER (Tỷ lệ thật → giả)')
    plt.plot(epochs, acers, 'k-o', label='ACER (Trung bình lỗi phân loại)')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Tỷ lệ lỗi', fontsize=12)
    plt.title('Các chỉ số lỗi phân loại của mô hình CDCN++', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'performance_metrics.png'), dpi=300)
    plt.show()

def plot_best_roc_curve(result_data):
    """Vẽ đường cong ROC của mô hình tốt nhất"""
    # Tìm mô hình có ACER thấp nhất
    best_model = min(result_data, key=lambda x: x['acer'])
    
    plt.figure(figsize=(8, 8))
    plt.plot(best_model['fpr'], best_model['tpr'], 'b-', linewidth=2,
             label=f'ROC curve (AUC = {best_model["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ Dương tính giả (FPR)', fontsize=12)
    plt.ylabel('Tỷ lệ Dương tính thật (TPR)', fontsize=12)
    plt.title(f'Đường cong ROC của mô hình CDCN++ tốt nhất (Epoch {best_model["epoch"]})', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Đánh dấu điểm tối ưu trên đường cong ROC
    optimal_idx = np.argmax(np.array(best_model['tpr']) - np.array(best_model['fpr']))
    optimal_fpr = best_model['fpr'][optimal_idx]
    optimal_tpr = best_model['tpr'][optimal_idx]
    
    plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=8,
             label=f'Điểm tối ưu (Ngưỡng = {best_model["threshold"]:.3f})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'best_roc_curve.png'), dpi=300)
    plt.show()

def display_best_model_stats(result_data):
    """Hiển thị thống kê chi tiết của mô hình tốt nhất"""
    # Tìm mô hình có ACER thấp nhất
    best_model = min(result_data, key=lambda x: x['acer'])
    
    print("\n=== THỐNG KÊ MÔ HÌNH CDCN++ TỐT NHẤT ===")
    print(f"Epoch: {best_model['epoch']}")
    print(f"Ngưỡng tối ưu: {best_model['threshold']:.4f}")
    print(f"Độ chính xác (ACC): {best_model['accuracy']:.2%}")
    print(f"Tỷ lệ lỗi phân loại giả → thật (APCER): {best_model['apcer']:.2%}")
    print(f"Tỷ lệ lỗi phân loại thật → giả (BPCER): {best_model['bpcer']:.2%}")
    print(f"Tỷ lệ lỗi phân loại trung bình (ACER): {best_model['acer']:.2%}")
    print(f"Diện tích dưới đường cong ROC (AUC): {best_model['auc']:.4f}")
    print("===========================================")
    
    # Tạo bảng dữ liệu để lưu vào file
    stats_df = pd.DataFrame([{
        'Metric': 'Epoch',
        'Value': best_model['epoch']
    }, {
        'Metric': 'Ngưỡng tối ưu',
        'Value': f"{best_model['threshold']:.4f}"
    }, {
        'Metric': 'Độ chính xác (ACC)',
        'Value': f"{best_model['accuracy']:.2%}"
    }, {
        'Metric': 'Tỷ lệ lỗi phân loại giả → thật (APCER)',
        'Value': f"{best_model['apcer']:.2%}"
    }, {
        'Metric': 'Tỷ lệ lỗi phân loại thật → giả (BPCER)',
        'Value': f"{best_model['bpcer']:.2%}"
    }, {
        'Metric': 'Tỷ lệ lỗi phân loại trung bình (ACER)',
        'Value': f"{best_model['acer']:.2%}"
    }, {
        'Metric': 'Diện tích dưới đường cong ROC (AUC)',
        'Value': f"{best_model['auc']:.4f}"
    }])
    
    # Lưu vào file CSV
    stats_df.to_csv(os.path.join(log_dir, 'best_model_stats.csv'), index=False)
    
    return best_model

def main():
    """Hàm chính để vẽ tất cả các biểu đồ và hiển thị kết quả"""
    print("Đang phân tích dữ liệu huấn luyện CDCN++...")
    
    # Đọc dữ liệu loss từ file log
    epochs_loss, absolute_losses, contrastive_losses = parse_train_log()
    
    if epochs_loss and absolute_losses and contrastive_losses:
        # Vẽ đồ thị loss
        plot_loss_curves(epochs_loss, absolute_losses, contrastive_losses)
    else:
        print("Không thể đọc dữ liệu loss từ file log.")
    
    # Đọc và phân tích dữ liệu đánh giá
    epochs_val, result_data = get_val_scores()
    
    if result_data:
        # Vẽ đồ thị các metrics theo thời gian
        plot_metrics(epochs_val, result_data)
        
        # Vẽ đường cong ROC của mô hình tốt nhất
        plot_best_roc_curve(result_data)
        
        # Hiển thị thống kê chi tiết của mô hình tốt nhất
        best_model = display_best_model_stats(result_data)
        
        print(f"\nĐã lưu tất cả biểu đồ và thống kê vào thư mục: {log_dir}")
    else:
        print("Không thể đọc dữ liệu đánh giá.")

if __name__ == "__main__":
    main()
