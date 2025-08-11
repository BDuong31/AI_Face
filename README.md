# AI-Face: Hệ Thống Điểm Danh Thông Minh Bằng Nhận Diện Khuôn Mặt

Một hệ thống điểm danh thông minh sử dụng công nghệ AI để nhận diện khuôn mặt và phát hiện giả mạo, được xây dựng với kiến trúc microservices.

## Kiến Trúc Hệ Thống

Hệ thống bao gồm 3 service chính:

```
AI-Face/
├── face_web/                    # 🌐 Web Service (API & Frontend)
├── face_recognition_service/    # 👤 Face Recognition Service
└── cdcn_service/               # 🛡️ Anti-Spoofing Service (CDCN++)
```

## Chi Tiết Các Service

### Face Web Service

**Công nghệ sử dụng:**
- Backend: Flask + FastAPI (Python)
- Database: PostgreSQL với Prisma ORM
- Frontend: HTML/CSS/JavaScript với Bootstrap
- WebSocket: Real-time communication
- Scheduling: APScheduler cho tự động hóa

**Tính năng chính:**
- Xác thực JWT cho sinh viên và giáo viên
- Dashboard riêng cho từng vai trò
- Quản lý lớp học và lịch học
- Điểm danh tự động theo lịch trình
- Điểm danh thủ công cho giáo viên
- Thông báo real-time qua WebSocket
- Quản lý phiên điểm danh với GPS validation
- Báo cáo và thống kê điểm danh

**API Endpoints:**
```
POST /api/student/login          # Đăng nhập sinh viên
POST /api/teacher/login          # Đăng nhập giáo viên
GET  /api/student/dashboard      # Dashboard sinh viên
GET  /api/teacher/dashboard      # Dashboard giáo viên
POST /api/teacher/classes/{id}/attendance/start  # Bắt đầu điểm danh
POST /api/teacher/classes/{id}/attendance/stop   # Kết thúc điểm danh
POST /api/student/attendance     # Sinh viên điểm danh
```

### Face Recognition Service

**Công nghệ sử dụng:**
- Deep Learning: FaceNet (TensorFlow)
- Face Detection: MTCNN (Multi-task CNN)
- Classification: SVM
- Framework: Flask

**Tính năng chính:**
- Phát hiện khuôn mặt với MTCNN (P-Net, R-Net, O-Net)
- Trích xuất đặc trưng khuôn mặt với FaceNet
- Nhận diện danh tính với SVM classifier
- Xử lý từ camera hoặc upload ảnh
- API RESTful cho integration

**Models:**
- **FaceNet**: Model pre-trained 20180402-114759.pb
- **MTCNN**: 3 networks cascade (det1.npy, det2.npy, det3.npy)
- **Classifier**: SVM model (facemodel.pkl)

**Workflow:**
1. **Face Detection**: MTCNN phát hiện khuôn mặt trong ảnh
2. **Face Alignment**: Căn chỉnh khuôn mặt theo landmarks
3. **Feature Extraction**: FaceNet tạo 128-dimensional embeddings
4. **Classification**: SVM phân loại danh tính

### CDCN Anti-Spoofing Service

**Công nghệ sử dụng:**
- Deep Learning: CDCN++ (Central Difference CNN)
- Framework: PyTorch
- Computer Vision: OpenCV

**Tính năng chính:**
- Phát hiện giả mạo khuôn mặt (Anti-Spoofing)
- Central Difference Convolution layers
- Spatial attention mechanisms
- Binary mask prediction
- Real-time processing

**Model Architecture:**
- **CDCN++**: Enhanced version với better feature extraction
- **Binary Mask**: Phân biệt vùng thật/giả
- **Spatial Attention**: Tập trung vào vùng quan trọng

## Cài Đặt và Chạy

### Yêu Cầu Hệ Thống
- Python 3.8+
- PostgreSQL 15+
- CUDA (optional, for GPU acceleration)
- Node.js 16+ (for Prisma)

### 1. Cài Đặt Database

```bash
# Chạy PostgreSQL với Docker
cd face_web
docker-compose up -d

# Hoặc cài đặt PostgreSQL thủ công và tạo database
createdb attendance_db
```

### 2. Cấu Hình Environment

```bash
# Tạo file .env trong face_web/
DATABASE_URL="postgresql://user_attendance:password123@localhost:5432/attendance_db"
SECRET_KEY="your-secret-key-here"
```

### 3. Chạy Face Web Service

```bash
cd face_web

# Cài đặt Python dependencies
pip install flask flask-cors prisma python-dotenv asyncio nest-asyncio jwt starlette apscheduler asgiref

# Cài đặt và generate Prisma client
npm install
npx prisma generate
npx prisma db push

# Chạy server
python app.py
```

Server chạy tại: http://localhost:8000

### 4. Chạy Face Recognition Service

```bash
cd face_recognition_service

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy service
cd src
python app.py
```

Service chạy tại: http://localhost:5001

### 5. Chạy CDCN Anti-Spoofing Service

```bash
cd cdcn_service

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy service
python app.py

gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:3000 --reload app:app

uvicorn app:app --host 0.0.0.0 --port 3000 --reload
```

Service chạy tại: http://localhost:5002

## Database Schema

```sql
-- Sinh viên
students (id, studentCode, name, password, createdAt, updatedAt)

-- Giáo viên  
teachers (id, teacherCode, name, password, createdAt, updatedAt)

-- Khóa học
courses (id, courseCode, courseName, description)

-- Lớp học
classes (id, classCode, semester, isActive, courseId, teacherId)

-- Ghi danh
enrollments (id, enrollmentDate, studentId, classId)

-- Lịch học
class_schedules (id, dayOfWeek, startTime, endTime, room, classId)

-- Phiên điểm danh
attendance_sessions (id, sessionDate, isOpen, openedAt, closedAt, classId)

-- Bản ghi điểm danh
attendance_records (id, attendanceTime, isPresent, latitude, longitude, sessionId, studentId)
```

## Testing

### Test Face Recognition
```bash
cd face_recognition_service
python src/face_rec_cam.py  # Test với camera
python src/face_rec.py --path video/test.mp4  # Test với video
```

### Test Anti-Spoofing
```bash
cd cdcn_service
python test_CDCN.py --gpu 0 --batchsize 9
```

## Training Models

### Train Face Recognition
```bash
cd face_recognition_service

# 1. Chuẩn bị dữ liệu (cắt và align faces)
python src/align_dataset_mtcnn.py Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32

# 2. Train classifier
python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000
```

### Train Anti-Spoofing
```bash
cd cdcn_service

# Train CDCN++ model
python train_CDCN_tensorboardx.py --gpu 0 --batchsize 9 --epochs 60 --lr 0.00008
```

## Performance Metrics

### Face Recognition
- **Accuracy**: ~95% trên dataset test
- **Processing Time**: ~200ms per face
- **False Accept Rate**: <1%

### Anti-Spoofing (CDCN++)
- **ACER**: 1.2%
- **EER**: 1.5%
- **Processing Time**: ~150ms per image

## API Documentation

### Face Recognition API
```bash
POST /recognize
Content-Type: multipart/form-data

Response:
{
  "success": true,
  "person_name": "Nguyen Van A",
  "confidence": 0.92,
  "processing_time": 0.18
}
```

### Anti-Spoofing API
```bash
POST /detect-spoofing
Content-Type: multipart/form-data

Response:
{
  "is_real": true,
  "confidence": 0.89,
  "score": 0.12
}
```

## Development

### Project Structure
```
AI-Face/
├── face_web/
│   ├── app.py                 # Main Flask+Starlette app
│   ├── prisma/schema.prisma   # Database schema
│   ├── templates/             # HTML templates
│   ├── static/                # CSS, JS, images
│   └── docker-compose.yaml    # PostgreSQL setup
├── face_recognition_service/
│   ├── src/
│   │   ├── app.py            # Flask API
│   │   ├── face_rec_cam.py   # Camera recognition
│   │   └── align/            # MTCNN models
│   ├── Models/               # Pre-trained models
│   └── Dataset/              # Training data
└── cdcn_service/
    ├── app.py                # Flask API
    ├── models/CDCNs.py       # CDCN++ architecture
    ├── train_CDCN.py         # Training script
    └── CDCNpp_BinaryMask_P1_07/  # Model checkpoints
```

## Changelog

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- FaceNet paper and implementation
- MTCNN for face detection
- CDCN++ for anti-spoofing
- PostgreSQL and Prisma ORM
- Flask and FastAPI frameworks

---




---------------
python -m venv venv
venv\Scripts\activate
pip freeze > requirements.txt
pip install -r requirements.txt


pnpx prisma studio
 localhost:5555 




------------------------- 
Chi tiết bước cài đặt database PostgreSQL và Prisma ORM:

cd face_web

# Dừng và xóa container hiện tại
docker-compose down

# Xóa volume để đảm bảo không còn dữ liệu cũ
docker volume rm face_web_postgres_data

# Khởi động lại container
docker-compose up -d

# Kiểm tra trạng thái container
docker ps

# Cài đặt phiên bản prisma cụ thể để tránh lỗi
npm install prisma@5.17.0

# Generate Prisma client
npx prisma@5.17.0 generate

# Push schema lên database
npx prisma@5.17.0 db push


# xem các bảng 
docker exec -it db_postgres psql -U user_attendance -d attendance_db -c "\dt"


Get-Content "c:\Users\ASUS\Desktop\AI_Face\face_web\fix_permissions.sql" | docker exec -i db_postgres psql -U user_attendance -d attendance_db



PS C:\Users\ASUS\Desktop\AI_Face\face_web> npx prisma studio
Environment variables loaded from .env
Prisma schema loaded from prisma\schema.prisma
Prisma Studio is up on http://localhost:5555


npx prisma migrate dev --name init

# rebuild docker containers
docker-compose down -v
docker-compose up --build



docker exec -it db_postgres psql -U user_attendance -d attendance_db
# Xem các bảng trong database
\dt

SELECT * FROM students;
# Xem dữ liệu trong bảng students

\d students
# Xem cấu trúc bảng students

python -m prisma generate

http://localhost:3000/

 cd face_recognition_service