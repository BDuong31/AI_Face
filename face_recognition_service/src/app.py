from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.disable_eager_execution() # Important to keep for your existing Facenet model

import argparse
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face # This is MTCNN
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import base64
from pathlib import Path
import subprocess

# --- Global Variables & Model Initialization ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Configuration (Adjust paths as needed) ---
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / 'Dataset' / 'FaceData' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'Dataset' / 'FaceData' / 'processed'
MODELS_DIR = BASE_DIR / 'Models'

ALIGN_SCRIPT = BASE_DIR / 'src/align_dataset_mtcnn.py'
CLASSIFIER_SCRIPT = BASE_DIR / 'src/classifier.py'
PB_MODEL_FILE = MODELS_DIR / '20180402-114759.pb'
PKL_MODEL_FILE = MODELS_DIR / 'facemodel.pkl'

print(BASE_DIR)
# Ensure these paths are correct relative to where you run the Flask app
CLASSIFIER_PATH = r'C:\Users\ASUS\Desktop\AI_Face\face_recognition_service\Models\facemodel.pkl'
FACENET_MODEL_PATH = r'C:\Users\ASUS\Desktop\AI_Face\face_recognition_service\Models\20180402-114759.pb'
# ### DEBUG ###: Verify this directory and its contents
MTCNN_MODEL_DIR = r'C:\Users\ASUS\Desktop\AI_Face\face_recognition_service\src\align'
print(f"### DEBUG ###: Kiểm tra thư mục: {MTCNN_MODEL_DIR}")
if not os.path.isdir(MTCNN_MODEL_DIR):
    print(f"### DEBUG ###: ERROR - MTCNN_MODEL_DIR không tồn tại hoặc không phải là thư mục!")
else:
    print(f"### DEBUG ###: Nội dung của MTCNN_MODEL_DIR: {os.listdir(MTCNN_MODEL_DIR)}")


# --- Load Models ---
print("Đang tải mô hình Facenet...")
tf_graph = tf.Graph()
with tf_graph.as_default():
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    tf_sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with tf_sess.as_default():
        facenet.load_model(FACENET_MODEL_PATH)
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        print("Mô hình Facenet đã được tải thành công.")

        print("Đang tải mô hình MTCNN...")
        try:
            pnet, rnet, onet = align.detect_face.create_mtcnn(tf_sess, MTCNN_MODEL_DIR)
            print("Mô hình MTCNN đã được tải thành công.")
        except Exception as e_mtcnn_load:
            print(f"### DEBUG ###: LỖI NGHIÊM TRỌNG khi tải MTCNN models: {e_mtcnn_load}")
            pnet, rnet, onet = None, None, None # Ensure they are None if loading failed

print(f"Đang tải mô hình phân loại từ {CLASSIFIER_PATH}...")
if not os.path.exists(CLASSIFIER_PATH):
    print(f"Error: Mô hình phân loại không tồn tại tại {CLASSIFIER_PATH}")
    sys.exit(1) # Or handle more gracefully if this is a non-critical part for pure detection
with open(CLASSIFIER_PATH, 'rb') as infile:
    model, class_names = pickle.load(infile)
print("Mô hình phân loại đã được tải thành công.")
print(f"Tên lớp: {class_names}")

# --- Flask API Endpoint ---
@app.route('/recognize', methods=['POST'])
def recognize_face():
    start_time = datetime.datetime.now() # For overall processing time measurement

    if 'image' not in request.files:
        return jsonify({"error": "Không có tệp hình ảnh nào được cung cấp"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Không có tệp nào được chọn"}), 400

    date_param = request.form.get('date', str(datetime.date.today()))
    class_id_param = request.form.get('classId', 'default_class')
    print(f"### DEBUG ###: Thông tin nhận được: {date_param}, classId: {class_id_param}")

    try:
        img_stream = file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("### DEBUG ###: cv2.imdecode trả về Không có. Không thể giải mã hình ảnh.")
            return jsonify({"error": "Không thể giải mã hình ảnh"}), 400

        print(f"### DEBUG ###: Hình ảnh đã được giải mã thành công. Kích thước khung: {frame.shape}, dtype: {frame.dtype}")

        # Check if MTCNN models loaded properly before trying to use them
        if pnet is None or rnet is None or onet is None:
            print("### DEBUG ###: Mô hình MTCNN chưa được tải. Không thể thực hiện phát hiện khuôn mặt.")
            return jsonify({"error": "Các mô hình MTCNN không được tải lên máy chủ", "details": "Không thể thực hiện phát hiện khuôn mặt."}), 500


        # --- Face Detection ---
        print(f"### DEBUG ###: Gọi align.detect_face.detect_face với MINSIZE={MINSIZE}, THRESHOLD={THRESHOLD}, FACTOR={FACTOR}")
        bounding_boxes, points = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        
        if bounding_boxes is None:
            print("### DEBUG ###: align.detect_face.detect_face trả về Không có cho bounding_boxes.")
            faces_found = 0
        else:
            faces_found = bounding_boxes.shape[0]
            print(f"### DEBUG ###: Các hộp giới hạn thô từ MTCNN (hình dạng: {bounding_boxes.shape}):\n{bounding_boxes}")
            if points is not None:
                 print(f"### DEBUG ###: Các điểm thô (landmarks) từ MTCNN (hình dạng: {points.shape}):\n{points}")
            else:
                 print("### DEBUG ###: MTCNN không trả về các điểm (landmarks).")


        print("### DEBUG ###: Số khuôn mặt phát hiện bởi MTCNN:", faces_found)

        results = []
        recognition_success = False # Overall success if at least one face is confidently recognized

        if faces_found > 0:
            det = bounding_boxes[:, 0:4] # x1, y1, x2, y2
            confidences = bounding_boxes[:, 4] # Confidence score for each bounding box

            bb = np.zeros((faces_found, 4), dtype=np.int32)

            for i in range(faces_found):
                print(f"### DEBUG ###: Đang xử lý khuôn mặt {i+1}/{faces_found}")
                print(f"### DEBUG ###:   MTCNN raw det[{i}]: {det[i]}")
                print(f"### DEBUG ###:   MTCNN confidence[{i}]: {confidences[i]}")

                # Ensure coordinates are within image bounds and valid
                bb[i][0] = np.maximum(int(det[i][0]), 0)
                bb[i][1] = np.maximum(int(det[i][1]), 0)
                bb[i][2] = np.minimum(int(det[i][2]), frame.shape[1])
                bb[i][3] = np.minimum(int(det[i][3]), frame.shape[0])
                print(f"### DEBUG ###:   Đã tính toán bb[{i}] (x1,y1,x2,y2): {bb[i]}")

                # Check for valid bounding box dimensions
                if bb[i][2] <= bb[i][0] or bb[i][3] <= bb[i][1]:
                    print(f"### DEBUG ###:   Bỏ qua khuôn mặt {i} do kích thước bbox không hợp lệ: w={bb[i][2]-bb[i][0]}, h={bb[i][3]-bb[i][1]}")
                    results.append({
                        "MSSV": "ErrorInvalidBBox",
                        "error": "Kích thước hộp giới hạn không hợp lệ từ MTCNN.",
                        "bbox_raw_mtcnn": [float(d) for d in det[i]],
                        "bbox_calculated": [int(b) for b in bb[i]]
                    })
                    continue
                
                try:
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    if cropped.size == 0:
                        print(f"### DEBUG ###:   Warning: Khu vực cắt cho khuôn mặt {i} là trống. BBox: {bb[i]}. Bỏ qua.")
                        results.append({
                            "MSSV": "ErrorEmptyCrop",
                            "error": "Khu vực cắt khuôn mặt trống.",
                            "bbox": [int(b) for b in bb[i]]
                        })
                        continue
                    print(f"### DEBUG ###:   Hình dạng khuôn mặt được cắt {i}: {cropped.shape}")

                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = tf_sess.run(embeddings, feed_dict=feed_dict)

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    
                    if not class_names: # Should not happen if classifier loaded
                        print("### DEBUG ###: class_names không có!")
                        best_name = "ErrorNoClassNames"
                    elif best_class_indices[0] >= len(class_names):
                        print(f"### DEBUG ###: best_class_indices[0] ({best_class_indices[0]}) là ngoài giới hạn cho class_names (len: {len(class_names)})")
                        best_name = "ErrorIndexOutOfBounds"
                    else:
                        best_name = class_names[best_class_indices[0]]


                    print(f"### DEBUG ###:   Khuôn mặt {i} - Đã phát hiện: {best_name}, Xác suất: {best_class_probabilities[0]:.4f}")

                    current_recognition = {
                        "MSSV": best_name,
                        "probability": float(best_class_probabilities[0]),
                        "bbox": [int(b) for b in bb[i]] # Bounding box [x1, y1, x2, y2]
                    }

                    if best_class_probabilities[0] > 0.70: # Confidence threshold for "known"
                        recognition_success = True # Mark overall success
                        # results.append(current_recognition) # Appended below
                    else:
                        current_recognition["MSSV"] = "Unknown" # If below threshold, classify as Unknown
                        current_recognition["original_prediction"] = best_name # Keep what it thought it was
                    
                    results.append(current_recognition)

                except Exception as e_face:
                    print(f"### DEBUG ###:   Lỗi xử lý khuôn mặt {i}: {e_face}")
                    results.append({
                        "MSSV": "ErrorInFaceProcessingLoop",
                        "error": str(e_face),
                        "bbox": [int(b) for b in bb[i]] if 'bb' in locals() and i < len(bb) else "N/A"
                    })
        
        end_time = datetime.datetime.now()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000

        response_payload = {
            "success": recognition_success,
            "message": "",
            "date": date_param,
            "classId": class_id_param,
            "recognitions": results,
            "faces_detected_count": faces_found,
            "processing_time_ms": round(processing_time_ms, 2)
        }

        if faces_found == 0:
            response_payload["message"] = "Không phát hiện khuôn mặt nào trong hình ảnh."
        elif not results and faces_found > 0: # Faces detected, but all resulted in errors or were skipped
             response_payload["message"] = "Đã phát hiện khuôn mặt, nhưng không thể nhận dạng hoặc xử lý bất kỳ khuôn mặt nào."
        elif not recognition_success and faces_found > 0: # Faces processed, but none met high confidence
            response_payload["message"] = "Quá trình nhận dạng đã hoàn tất, nhưng không ai được xác định với độ tin cậy cao."
        elif recognition_success:
            response_payload["message"] = "Quá trình nhận dạng đã hoàn tất."

        return jsonify(response_payload), 200

    except Exception as e:
        print(f"### DEBUG ###: Lỗi nghiêm trọng trong /recognize endpoint: {e}")
        import traceback
        traceback.print_exc() # Print full traceback to Flask console
        return jsonify({"error": "Đã xảy ra lỗi máy chủ nội bộ", "details": str(e)}), 500

@app.route('/api/student/exists/<string:student_id>', methods=['GET'])
def check_student_exists(student_id):
    """
    Kiểm tra xem dữ liệu khuôn mặt của một sinh viên (dựa trên MSSV)
    đã tồn tại trong mô hình phân loại đã được huấn luyện hay chưa.
    """
    try:
        # `class_names` là danh sách các MSSV được tải từ file model .pkl
        # Thao tác này rất nhanh vì nó chỉ kiểm tra sự tồn tại trong một danh sách
        student_exists = student_id in class_names

        if student_exists:
            message = "Dữ liệu sinh viên đã tồn tại trong mô hình nhận diện."
        else:
            message = "Không tìm thấy dữ liệu sinh viên trong mô hình nhận diện."

        response_payload = {
            "student_id": student_id,
            "exists": student_exists,
            "message": message
        }
        
        # Trả về 200 OK trong cả hai trường hợp vì yêu cầu đã được xử lý thành công.
        # Giá trị của 'exists' cho biết kết quả của việc kiểm tra.
        return jsonify(response_payload), 200

    except Exception as e:
        print(f"### DEBUG ###: Lỗi trong endpoint /api/student/exists: {e}")
        return jsonify({"error": "Lỗi server nội bộ", "details": str(e)}), 500

@app.route('/add_student_and_train', methods=['POST'])
def add_student_and_train():
    """
    API nhận thông tin sinh viên mới, lưu ảnh, tiền xử lý và huấn luyện lại mô hình.
    """
    # --- BƯỚC 0: NHẬN VÀ KIỂM TRA DỮ LIỆU ---
    try:
        data = request.get_json()
        if not data or 'studentId' not in data or 'images' not in data:
            return jsonify({"success": False, "message": "Dữ liệu không hợp lệ. Cần có 'studentId' và 'images'."}), 400
        
        student_id = data['studentId']
        images_base64 = data['images']
        print(f"Nhận được yêu cầu cho sinh viên: {student_id} với {len(images_base64)} ảnh.")

    except Exception as e:
        return jsonify({"success": False, "message": f"Lỗi khi đọc dữ liệu JSON: {e}"}), 400

    # --- BƯỚC 1: LƯU ẢNH ---
    student_raw_dir = RAW_DATA_DIR / str(student_id)
    try:
        if not os.path.exists(student_raw_dir):
            os.makedirs(student_raw_dir)
            print(f"Đã tạo thư mục: {student_raw_dir}")

        for i, image_b64 in enumerate(images_base64):
            # Tách phần tiền tố 'data:image/jpeg;base64,'
            image_data = image_b64.split(",")[1]
            file_path = os.path.join(student_raw_dir, f"{student_id}_{i + 1}.jpg")
            with open(file_path, "wb") as fh:
                fh.write(base64.b64decode(image_data))
        print(f"Đã lưu thành công {len(images_base64)} ảnh cho sinh viên {student_id}.")
    except Exception as e:
        print(f"Lỗi khi lưu ảnh cho {student_id}: {e}")
        return jsonify({"success": False, "step": "save_images", "message": f"Lỗi khi lưu ảnh: {e}"}), 500

    # --- BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU ---
    try:
        print("Bắt đầu tiền xử lý dữ liệu...")
        preprocess_command = [
            'python', ALIGN_SCRIPT, 
            RAW_DATA_DIR, PROCESSED_DATA_DIR, 
            '--image_size', '160', 
            '--margin', '32', 
            '--random_order', 
            '--gpu_memory_fraction', '0.25'
        ]
        # Chạy lệnh và chờ hoàn tất, nếu có lỗi sẽ văng exception
        result = subprocess.run(preprocess_command, check=True, capture_output=True, text=True)
        print("Tiền xử lý dữ liệu thành công.")
        print(f"Output tiền xử lý:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi trong quá trình tiền xử lý: {e}")
        # Trả về lỗi chi tiết từ script python
        return jsonify({"success": False, "step": "preprocess", "message": f"Lỗi tiền xử lý: {e.stderr}"}), 500

    # --- BƯỚC 3: HUẤN LUYỆN LẠI MÔ HÌNH ---
    try:
        print("Bắt đầu huấn luyện lại mô hình...")
        train_command = [
            'python', CLASSIFIER_SCRIPT, 
            'TRAIN', PROCESSED_DATA_DIR, 
            PB_MODEL_FILE, PKL_MODEL_FILE, 
            '--batch_size', '1000'
        ]
        result = subprocess.run(train_command, check=True, capture_output=True, text=True)
        print("Huấn luyện mô hình thành công.")
        print(f"Output huấn luyện:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi trong quá trình huấn luyện: {e}")
        return jsonify({"success": False, "step": "train", "message": f"Lỗi huấn luyện: {e.stderr}"}), 500

    # --- BƯỚC 4: THÀNH CÔNG ---
    return jsonify({
        "success": True, 
        "message": f"Hoàn tất quá trình thêm và huấn luyện cho sinh viên {student_id}."
    })

if __name__ == '__main__':
    print("### DEBUG ###: Khởi động ứng dụng Flask cho dịch vụ nhận dạng...")
    # TODO: Consider adding a warm-up call here if possible/needed
    # For example, load a dummy image and send it to the /recognize endpoint internally
    # This might be complex depending on how you structure it without an actual HTTP client.
    # A simpler approach for TF specific warm-up is to run a dummy inference right after model loading.

    # Example of a simple TensorFlow warm-up after models are loaded and BEFORE app.run()
    # This is more direct than an HTTP call for warming up TF itself.
    with tf_graph.as_default():
        with tf_sess.as_default():
            if pnet and rnet and onet and images_placeholder is not None: # Check if models loaded
                print("### DEBUG ###: Thực hiện khởi động TensorFlow...")
                try:
                    # Create a dummy image tensor of the expected input size
                    dummy_image_np = np.zeros((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3), dtype=np.float32)
                    dummy_image_np = facenet.prewhiten(dummy_image_np)
                    dummy_image_reshaped = dummy_image_np.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                    # Warm-up Facenet
                    _ = tf_sess.run(embeddings, feed_dict={
                        images_placeholder: dummy_image_reshaped,
                        phase_train_placeholder: False
                    })

                    # Warm-up MTCNN (requires a frame-like input)
                    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Example frame size
                    _ = align.detect_face.detect_face(dummy_frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                    print("### DEBUG ###: Khởi động TensorFlow đã hoàn tất.")
                except Exception as e_warmup:
                    print(f"### DEBUG ###: Lỗi trong quá trình khởi động TensorFlow: {e_warmup}")

    app.run(host='0.0.0.0', port=5001, debug=False, threaded=False)