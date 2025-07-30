from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# Tương thích với TensorFlow 1.x
tf.compat.v1.disable_eager_execution()
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tensorboardX import SummaryWriter 

# <<< THÊM MỚI: Thư viện để xử lý ảnh cho TensorBoard >>>
# Cần cài đặt: pip install torch torchvision
# Mặc dù code dùng TF, nhưng torchvision có tiện ích tạo lưới ảnh rất tốt
try:
    import torch
    from torchvision.utils import make_grid
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Cảnh báo: Không tìm thấy PyTorch/Torchvision. Tính năng hiển thị lưới ảnh sẽ bị tắt.")
    print("Để bật, hãy cài đặt bằng lệnh: pip install torch torchvision")


def main(args):
    # Khởi tạo SummaryWriter để ghi log cho TensorBoard
    # Tạo một thư mục log riêng cho mỗi lần chạy dựa trên tên file classifier
    log_path = os.path.join(args.log_dir, os.path.basename(args.classifier_filename).split('.')[0])
    writer = SummaryWriter(log_dir=log_path)
    print(f"TensorBoard log sẽ được lưu tại: {log_path}")

    # <<< THÊM MỚI: Ghi lại các tham số đã sử dụng >>>
    writer.add_text('Parameters', str(vars(args)), 0)
  
    with tf.Graph().as_default():
      
        with tf.compat.v1.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')

            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # <<< THÊM MỚI: Ghi số liệu thống kê cơ bản >>>
            writer.add_scalar('Dataset/Number_of_classes', len(dataset), 0)
            writer.add_scalar('Dataset/Number_of_images', len(paths), 0)

            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]

            if (args.mode=='TRAIN'):
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

                # --- Bổ sung cho TensorBoard (TRAIN) ---
                print('Writing training data to TensorBoard...')
                
                # 1. Trực quan hóa Embeddings (như cũ)
                metadata_labels = [class_names[i] for i in labels]
                images_for_projector = facenet.load_data(paths, False, False, args.image_size)
                if images_for_projector.ndim == 4:
                    images_for_projector = images_for_projector.transpose(0, 3, 1, 2)
                
                writer.add_embedding(
                    mat=emb_array,
                    metadata=metadata_labels,
                    label_img=images_for_projector,
                    tag='Train_Embeddings'
                )

                # 2. <<< THÊM MỚI: Histogram của các giá trị embedding >>>
                # Giúp xem phân phối của các vector đặc trưng
                writer.add_histogram('Train/Embedding_Distribution', emb_array, 0)

                # 3. <<< THÊM MỚI: Hiển thị một vài ảnh train mẫu >>>
                if TORCH_AVAILABLE:
                    sample_images_paths = paths[:min(32, len(paths))]
                    sample_images = facenet.load_data(sample_images_paths, False, False, args.image_size)
                    sample_images_tensor = torch.from_numpy(sample_images).permute(0, 3, 1, 2)
                    grid = make_grid(sample_images_tensor, normalize=True)
                    writer.add_image('Train/Sample_Images', grid, 0)

                print('Done writing to TensorBoard.')
                
            elif (args.mode=='CLASSIFY'):
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = accuracy_score(labels, best_class_indices)
                print('Accuracy: %.3f' % accuracy)

                # --- Bổ sung cho TensorBoard (CLASSIFY) ---
                print('Writing classification results to TensorBoard...')
                
                # 1. <<< NÂNG CẤP: Ghi các chỉ số Scalar >>>
                # Ghi lại Accuracy, Precision, Recall, F1-score
                prec, recall, f1, _ = precision_recall_fscore_support(labels, best_class_indices, average='macro')
                writer.add_scalar('Metrics/Accuracy', accuracy, 0)
                writer.add_scalar('Metrics/Precision_Macro', prec, 0)
                writer.add_scalar('Metrics/Recall_Macro', recall, 0)
                writer.add_scalar('Metrics/F1_Score_Macro', f1, 0)

                # 2. <<< THÊM MỚI: Ghi báo cáo phân loại dạng text >>>
                report = classification_report(labels, best_class_indices, target_names=class_names)
                # Chuyển đổi báo cáo thành định dạng thân thiện với Markdown cho TensorBoard
                formatted_report = '```\n' + report + '\n```'
                writer.add_text('Metrics/Classification_Report', formatted_report, 0)

                # 3. <<< THÊM MỚI: Histogram về độ chắc chắn của dự đoán >>>
                writer.add_histogram('Test/Prediction_Probabilities', best_class_probabilities, 0)

                # 4. Trực quan hóa Embeddings của tập test (như cũ nhưng với tag khác)
                metadata_labels = [f"True: {class_names[l]}, Pred: {class_names[p]}" for l, p in zip(labels, best_class_indices)]
                images_for_projector = facenet.load_data(paths, False, False, args.image_size)
                if images_for_projector.ndim == 4:
                    images_for_projector = images_for_projector.transpose(0, 3, 1, 2)

                writer.add_embedding(
                    mat=emb_array,
                    metadata=metadata_labels,
                    label_img=images_for_projector,
                    tag='Test_Embeddings'
                )

                # 5. <<< THÊM MỚI: Trực quan hóa các ảnh bị phân loại sai >>>
                if TORCH_AVAILABLE:
                    misclassified_indices = np.where(best_class_indices != labels)[0]
                    if len(misclassified_indices) > 0:
                        # Lấy tối đa 32 ảnh bị phân loại sai
                        misclassified_paths = [paths[i] for i in misclassified_indices[:32]]
                        misclassified_images = facenet.load_data(misclassified_paths, False, False, args.image_size)
                        
                        true_labels_text = [class_names[labels[i]] for i in misclassified_indices[:32]]
                        pred_labels_text = [class_names[best_class_indices[i]] for i in misclassified_indices[:32]]
                        
                        # Tạo caption cho mỗi ảnh
                        captions = [f"True: {t}\nPred: {p}" for t, p in zip(true_labels_text, pred_labels_text)]

                        # Chuyển đổi sang tensor và tạo lưới
                        misclassified_tensor = torch.from_numpy(misclassified_images).permute(0, 3, 1, 2)
                        grid = make_grid(misclassified_tensor, normalize=True, nrow=8) # Hiển thị 8 ảnh mỗi hàng
                        writer.add_image('Test/Misclassified_Images', grid, 0)
                        writer.add_text('Test/Misclassified_Captions', " | ".join(captions), 0)

                print('Done writing to TensorBoard.')
    
    writer.close()
    print("Đã đóng TensorBoard writer. Dữ liệu đã được lưu.")
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    # (Hàm này giữ nguyên không đổi)
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set
            
def parse_arguments(argv):
    # (Hàm này giữ nguyên không đổi)
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    
    parser.add_argument('--log_dir', type=str,
        help='Directory where to write event logs for TensorBoard.', default='logs/')
        
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))