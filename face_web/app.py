import os
import jwt
import nest_asyncio # THÊM VÀO: Import nest_asyncio
import base64
import asyncio
import requests
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, jsonify, render_template, request, g
from flask_cors import CORS
from prisma import Prisma
from dotenv import load_dotenv
from asgiref.wsgi import WsgiToAsgi
import json
from starlette.applications import Starlette
from starlette.websockets import WebSocket
from starlette.routing import Mount, WebSocketRoute
from math import radians, cos, sin, asin, sqrt
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# lấy giờ theo múi giờ Việt Nam
scheduler = AsyncIOScheduler(timezone="Asia/Ho_Chi_Minh")

# Quản lý kết nối WebSocket
class ConnectionManager:

    # Khởi tạo kết nối WebSocket
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    # Kết nối WebSocket cho giáo viên
    async def connect(self, websocket: WebSocket, teacher_id: str):
        await websocket.accept()
        self.active_connections[teacher_id] = websocket
        print(f"INFO:     Giáo viên {teacher_id} đã kết nối WebSocket.")

    # Ngắt kết nối WebSocket cho giáo viên
    def disconnect(self, teacher_id: str):
        if teacher_id in self.active_connections:
            del self.active_connections[teacher_id]
            print(f"INFO:     Giáo viên {teacher_id} đã ngắt kết nối WebSocket.")

    # Gửi tin nhắn cá nhân đến giáo viên qua WebSocket
    async def send_personal_message(self, message: dict, teacher_id: str):
        """
        Gửi tin nhắn cá nhân đến giáo viên qua WebSocket.
            :param message: Tin nhắn cần gửi, dạng dict.
            :param teacher_id: ID của giáo viên nhận tin nhắn.
        """

        # Kiểm tra xem giáo viên có kết nối WebSocket không
        if teacher_id in self.active_connections:
            websocket = self.active_connections[teacher_id] # Lấy kết nối WebSocket của giáo viên
            await websocket.send_text(json.dumps(message)) # Gửi tin nhắn dưới dạng JSON
            print(f"INFO:     Đã gửi thông báo cho giáo viên {teacher_id}.")

# Tạo một instance của ConnectionManager để quản lý kết nối WebSocket
manager = ConnectionManager()

# THÊM VÀO: Sửa lỗi asyncio event loop đã được chạy
nest_asyncio.apply()

# THÊM VÀO: Tải biến môi trường từ file .env
load_dotenv()

# THÊM VÀO: Tạo lớp LifespanApp để quản lý sự kiện khởi động và tắt ứng dụng
class LifespanApp:
    """
        Lớp này quản lý sự kiện khởi động và tắt ứng dụng ASGI.
        Nó cho phép thực hiện các tác vụ như kết nối cơ sở dữ liệu khi ứng dụng khởi động
        và ngắt kết nối khi ứng dụng tắt.
        Sử dụng lớp này để bao bọc ứng dụng Flask hoặc Starlette của bạn.
    """

    # Khởi tạo LifespanApp với ứng dụng ASGI và các hàm callback cho sự kiện khởi động và tắt
    def __init__(self, app, on_startup=None, on_shutdown=None):
        self.app = app
        self.on_startup = on_startup
        self.on_shutdown = on_shutdown

    # Hàm __call__ để xử lý các sự kiện lifespan
    async def __call__(self, scope, receive, send):
        """
            Hàm này được gọi khi ứng dụng ASGI nhận được yêu cầu.
            Nếu scope là 'lifespan', nó sẽ xử lý các sự kiện khởi động và tắt ứng dụng.
            Nếu không, nó sẽ chuyển tiếp yêu cầu đến ứng dụng ASGI đã được cung cấp.
            :param scope: Thông tin về yêu cầu ASGI.
            :param receive: Hàm nhận dữ liệu từ kết nối ASGI.
            :param send: Hàm gửi dữ liệu đến kết nối ASGI.
        """

        # Kiểm tra xem scope có phải là 'lifespan' không
        if scope['type'] == 'lifespan':
            # Nếu là 'lifespan', bắt đầu vòng lặp để nhận và gửi các sự kiện
            while True:
                message = await receive() # Nhận thông điệp từ kết nối ASGI
                if message['type'] == 'lifespan.startup': # Nếu là sự kiện khởi động
                    if self.on_startup: # Nếu có hàm khởi động, gọi nó
                        await self.on_startup() 
                    await send({'type': 'lifespan.startup.complete'}) # Gửi thông báo hoàn thành khởi động
                elif message['type'] == 'lifespan.shutdown': # Nếu là sự kiện tắt ứng dụng
                    if self.on_shutdown: # Nếu có hàm tắt ứng dụng, gọi nó
                        await self.on_shutdown()
                    await send({'type': 'lifespan.shutdown.complete'}) # Gửi thông báo hoàn thành tắt ứng dụng
                    return
        else:
            # Nếu không phải là 'lifespan', chuyển tiếp yêu cầu đến ứng dụng ASGI đã được cung cấp
            await self.app(scope, receive, send)

# THÊM VÀO: Khởi tạo Prisma ORM
prisma = Prisma(auto_register=True)

# --- Sự kiện khởi động và tắt ứng dụng ---
async def startup_event():
    """Kết nối tới cơ sở dữ liệu khi server khởi động."""
    print("INFO:     Đang kết nối tới cơ sở dữ liệu...")
    if not prisma.is_connected():
        await prisma.connect()
    print("INFO:     Đã kết nối cơ sở dữ liệu thành công.")
    
    await schedule_weekly_attendance_jobs()

    scheduler.start()

    print("INFO:     ✅ Scheduler started and jobs are scheduled.")

# --- Sự kiện tắt ứng dụng ---
async def shutdown_event():
    """Ngắt kết nối cơ sở dữ liệu khi server tắt."""
    print("INFO:     Shutting down the scheduler...")
    scheduler.shutdown()
    print("INFO:     Đang ngắt kết nối cơ sở dữ liệu...")
    if prisma.is_connected():
        await prisma.disconnect()
    print("INFO:     Đã đóng kết nối cơ sở dữ liệu.")

# --- Khởi tạo ứng dụng Flask ---
flask_app = Flask(__name__, static_folder='static', static_url_path='/static')
flask_app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key-for-dev')
CORS(flask_app)

# --- Đường dẫn WebSocket ---
async def websocket_endpoint(websocket: WebSocket):
    # Lấy teacher_id từ URL, ví dụ: /ws/some_teacher_id
    teacher_id = websocket.path_params['teacher_id']
    await manager.connect(websocket, teacher_id)
    try:
        while True:
            # Giữ kết nối mở để lắng nghe
            # Chúng ta không cần nhận dữ liệu từ client ở đây, nhưng vòng lặp là cần thiết
            await websocket.receive_text()
    except Exception:
        manager.disconnect(teacher_id)

# --- Khởi tạo ứng dụng Starlette ---
app = Starlette(
    debug=True,
    routes=[
        # Route cho WebSocket
        WebSocketRoute("/ws/{teacher_id}", websocket_endpoint),
        # Gắn ứng dụng Flask để xử lý tất cả các route HTTP còn lại
        Mount("/", app=WsgiToAsgi(flask_app))
    ],
    on_startup=[startup_event],
    on_shutdown=[shutdown_event],
)

# --- Decorators cho xác thực token cho sinh viên ---
def token_required(f):
    @wraps(f) 

    # Decorator để yêu cầu token xác thực cho các route
    async def decorated(*args, **kwargs): 
        token = None # Biến để lưu token từ header

        # Kiểm tra xem có header Authorization không
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]

        # Nếu không có token, trả về lỗi 401
        if not token:
            return jsonify({'message': 'Token không tồn tại!'}), 401

        try:
            # Giải mã token và lấy dữ liệu
            data = jwt.decode(token, flask_app.config['SECRET_KEY'], algorithms=["HS256"])

            # Tìm sinh viên dựa trên student_id trong token
            current_student = await prisma.student.find_unique(where={'id': data['student_id']})

            # Nếu không tìm thấy sinh viên, trả về lỗi 404
            if not current_student:
                return jsonify({'message': 'Không tìm thấy sinh viên!'}), 404
            
            # Lưu thông tin sinh viên vào `g` object, có thể truy cập trong suốt request
            g.current_student = current_student

        # Xử lý các lỗi khi giải mã token
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token đã hết hạn!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token không hợp lệ!'}), 401
        
        # Gọi hàm gốc mà không truyền thêm tham số
        return await f(*args, **kwargs)

    return decorated

# Decorator để yêu cầu token xác thực cho giáo viên
def teacher_token_required(f):
    @wraps(f)

    # Decorator để yêu cầu token xác thực cho các route của giáo viên
    async def decorated(*args, **kwargs):
        token = None # Biến để lưu token từ header

        # Kiểm tra xem có header Authorization không
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]

        # Nếu không có token, trả về lỗi 401
        if not token:
            return jsonify({'message': 'Token không tồn tại!'}), 401

        try:
            # Giải mã token và lấy dữ liệu
            data = jwt.decode(token, flask_app.config['SECRET_KEY'], algorithms=["HS256"])
            
            # Tìm giáo viên dựa trên teacher_id trong token
            current_teacher = await prisma.teacher.find_unique(where={'id': data['teacher_id']})
            
            # Nếu không tìm thấy giáo viên, trả về lỗi 404
            if not current_teacher:
                return jsonify({'message': 'Không tìm thấy giáo viên!'}), 404
            
            # Lưu thông tin giáo viên vào `g` object, có thể truy cập trong suốt request
            g.current_teacher = current_teacher

        # Xử lý các lỗi khi giải mã token
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token đã hết hạn!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token không hợp lệ!'}), 401
        
        # Gọi hàm gốc mà không truyền thêm tham số
        return await f(*args, **kwargs)

    return decorated

# --- Các hàm tiện ích ---

# --- Kiểm tra xem ảnh có phải là khuôn mặt thật hay không (Liveness) ---
def is_real_face(image_data_full):
    """
        Gửi ảnh đến API phát hiện thật/giả (localhost:5002) dưới dạng form-data với key là 'image_data'.
    """
    liveness_api_url = "http://127.0.0.1:5002/predict"

    # Chuẩn bị payload dạng form-data với key là 'image_data'
    payload = {"image_data": image_data_full}
    
    try:
        print("--- Đang gọi API phát hiện thật/giả (Liveness) với form-data...")
        
        # Gửi request với payload là data (form-data)
        response = requests.post(liveness_api_url, data=payload, timeout=10)
        
        response.raise_for_status() 
        
        result = response.json()
        print(f"Kết quả từ Liveness API: {result}")

        # API của bạn trả về 'label', kiểm tra key này
        if result.get("label") == "Thật (Real)":
            return True
        else:
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"LỖI khi gọi Liveness API: {e}")
        return False
    except Exception as e:
        print(f"LỖI không xác định trong is_real_face: {e}")
        return False

# --- Nhận diện khuôn mặt từ ảnh ---
def recognize_face(image_data_full, student_code):
    """
    Gửi ảnh đến API Nhận diện (localhost:5001) dưới dạng multipart/form-data
    với key là 'image' và xử lý kết quả chi tiết.
    """
    recognition_api_url = "http://127.0.0.1:5001/recognize"
    
    try:
        raw_image_data = image_data_full.split(',', 1)[1]
        image_bytes = base64.b64decode(raw_image_data)
        
        files = {'image': ('recognition_image.jpg', image_bytes, 'image/jpeg')}
        
        print("--- Đang gọi API nhận diện với multipart/form-data...")
        
        # 3. Gửi request
        response = requests.post(recognition_api_url, files=files, timeout=15) # Tăng timeout cho nhận diện
        response.raise_for_status()

        result = response.json()
        print(f"Kết quả từ Recognition API: {result}")

        # 4. Xử lý kết quả
        if result.get("success"):
            recognitions = result.get("recognitions", [])
            for person in recognitions:
                # Trả về MSSV của người đầu tiên được nhận diện không phải là "Unknown"
                if person.get("MSSV") and person.get("MSSV") != "Unknown":
                    # Kiểm tra xem MSSV có khớp với MSSV của sinh viên hiện tại không
                    if person.get("MSSV") == student_code:
                        return person.get("MSSV")
            # Nếu không có ai được nhận diện thành công
        return None

    except requests.exceptions.RequestException as e:
        print(f"LỖI khi gọi Recognition API: {e}")
        return None
    except (base64.binascii.Error, ValueError, IndexError) as e:
        print(f"LỖI decode base64 trong recognize_face: {e}")
        return None
    except Exception as e:
        print(f"LỖI không xác định trong recognize_face: {e}")
        return None

# --- Kiểm tra IP có nằm trong danh sách cho phép ---
def check_ip_allowed(ip_address=None):
    """
    Kiểm tra xem IP có nằm trong danh sách cho phép hay không.
    Nếu không truyền ip_address, sẽ kiểm tra IP của request hiện tại.
    """
    # allowed_ips = "113.162.208.236"
    allowed_ips = "27.66.20.115" # IP nhà Quí
    if not ip_address:
        ip_address = request.remote_addr  

    print(f"INFO:     Đang kiểm tra IP: {ip_address}...")

    if ip_address in allowed_ips:
        print("INFO:     IP hợp lệ.")
        return True
    else:
        print("INFO:     IP không hợp lệ.")
        return False

# --- Tính khoảng cách giữa hai điểm (kinh độ, vĩ độ) ---
def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Tính khoảng cách giữa hai điểm (kinh độ, vĩ độ) trên Trái Đất bằng km.
    """
    # Chuyển đổi độ sang radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Công thức Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Bán kính Trái Đất tính bằng km
    return c * r

# --- Tác vụ tự động mở và đóng phiên điểm danh ---
async def auto_open_attendance_session(class_id: str):
    """Tác vụ tự động mở phiên điểm danh."""
    async with Prisma() as db:
        print(f"LỊCH TRÌNH: [CHẠY CÔNG VIỆC] Kích hoạt tự động mở cho lớp học: {class_id}")
        existing_open_session = await db.attendancesession.find_first(where={'classId': class_id, 'isOpen': True})
        if existing_open_session:
            print(f"LỊCH TRÌNH: [CHẠY CÔNG VIỆC] Phiên điểm danh cho lớp {class_id} đã mở. Bỏ qua việc mở tự động.")
            return

        cls = await db.courseclass.find_unique(where={'id': class_id}, include={'enrollments': True})
        if not cls:
            print(f"LỊCH TRÌNH: [CHẠY CÔNG VIỆC] Lớp học {class_id} không tồn tại.")
            return

        new_session = await db.attendancesession.create(data={'classId': class_id, 'sessionDate': datetime.now(), 'isOpen': True, 'openedAt': datetime.now()})
        if cls.enrollments:
            records_to_create = [{'sessionId': new_session.id, 'studentId': e.studentId, 'status': 'UNMARKED'} for e in cls.enrollments]
            await db.attendancerecord.create_many(data=records_to_create)
        print(f"LỊCH TRÌNH: ✅ Phiên mở tự động thành công {new_session.id} cho lớp học {class_id}.")

# --- Tác vụ tự động đóng phiên điểm danh ---
async def auto_close_attendance_session_by_class_id(class_id: str):
    """Tác vụ tự động đóng phiên điểm danh."""
    async with Prisma() as db:
        print(f"LỊCH TRÌNH: [CHẠY CÔNG VIỆC] Kích hoạt tự động đóng cho lớp: {class_id}")
        open_session = await db.attendancesession.find_first(where={'classId': class_id, 'isOpen': True})
        if not open_session:
            print(f"LỊCH TRÌNH: [CHẠY CÔNG VIỆC] Không tìm thấy phiên mở cho lớp {class_id}. Bỏ qua việc đóng tự động.")
            return
            
        await db.attendancesession.update(where={'id': open_session.id}, data={'isOpen': False, 'closedAt': datetime.now()})
        await db.attendancerecord.update_many(where={'sessionId': open_session.id, 'status': 'UNMARKED'}, data={'status': 'ABSENT'})
        print(f"LỊCH TRÌNH: ✅ Đóng tự động phiên {open_session.id} thành công.")

# --- Tác vụ quét và lên lịch các tác vụ điểm danh hàng tuần ---
async def schedule_weekly_attendance_jobs():
    """Quét DB và lên lịch các tác vụ hàng tuần."""
    print("Lịch học: Đang quét và lên lịch các tác vụ điểm danh tự động...")
    scheduler.remove_all_jobs()
    all_schedules = await prisma.classschedule.find_many() # Dùng prisma toàn cục để đọc lịch trình
    
    for schedule in all_schedules:
        try:
            start_hour = 0
            start_minute = 0

            # --- FIX: Xử lý linh hoạt mọi định dạng dữ liệu ---
            if isinstance(schedule.startTime, datetime):
                # TRƯỜNG HỢP 1: Dữ liệu là đối tượng DateTime
                start_hour = schedule.startTime.hour
                start_minute = schedule.startTime.minute

            elif isinstance(schedule.startTime, str):
                # TRƯỜNG HỢP 2: Dữ liệu là chuỗi (String)
                time_part = schedule.startTime.strip()
                
                # Nếu chuỗi chứa dấu cách, đó là ngày giờ đầy đủ (VD: "2024-07-30 07:30:00")
                if ' ' in time_part:
                    # Tách lấy phần thời gian "07:30:00"
                    time_part = time_part.split(' ')[1] 
                
                # Bây giờ xử lý phần thời gian (VD: "07:30" hoặc "07:30:00")
                time_components = time_part.split(':')
                start_hour = int(time_components[0])
                start_minute = int(time_components[1])
            
            else:
                # Báo lỗi nếu gặp kiểu dữ liệu không mong muốn
                raise TypeError(f"Kiểu dữ liệu không được hỗ trợ cho startTime: {type(schedule.startTime)}")
            # --- KẾT THÚC SỬA LỖI ---


            open_time = (datetime.now().replace(hour=start_hour, minute=start_minute) - timedelta(minutes=15))
            close_time = (datetime.now().replace(hour=start_hour, minute=start_minute) + timedelta(minutes=15))
            
            open_job_id = f"auto_open_{schedule.id}"
            close_job_id = f"auto_close_{schedule.id}"

            scheduler.add_job(auto_open_attendance_session, 'cron', day_of_week=schedule.dayOfWeek, hour=open_time.hour, minute=open_time.minute, id=open_job_id, args=[schedule.classId], replace_existing=True)
            scheduler.add_job(auto_close_attendance_session_by_class_id, 'cron', day_of_week=schedule.dayOfWeek, hour=close_time.hour, minute=close_time.minute, id=close_job_id, args=[schedule.classId], replace_existing=True)
        except Exception as e:
            print(f"LỊCH TRÌNH: ERROR - Không thể lên lịch công việc cho ID lịch {schedule.id}. Lý do: {e}")
    print(f"LỊCH TRÌNH: Hoàn tất việc lên lịch công việc. Tổng số công việc: {len(scheduler.get_jobs())}")
    for job in scheduler.get_jobs():
        print(f"LỊCH TRÌNH: Công việc đã lên lịch: {job}")

# --- Các Route Giao Diện ---

# Trang landing page
@flask_app.route('/')
async def index():
    return render_template('landingpage.html') 

# Trang đăng nhập cho sinh viên
@flask_app.route('/student/login')
async def student_login_page():
    return render_template('loginStudent.html')

# Trang đăng nhập cho giáo viên
@flask_app.route('/teacher/login')
async def teacher_login_page():
    return render_template('loginTeacher.html')

# Trang quản lý điểm danh cho sinh viên
@flask_app.route('/student/dashboard')
async def student_dashboard_page():
    return render_template('studentdashboard.html')

# Trang quản lý điểm danh cho giáo viên
@flask_app.route('/teacher/dashboard')
async def teacher_dashboard_page():
    return render_template('teacherdashboard.html')

# Trang thu thập khuôn mặt cho sinh viên
@flask_app.route('/add-face-student')
async def add_face_student_page():
    return render_template('addFaceStudent.html')

# --- API Xác Thực ---

# API đăng nhập cho sinh viên
@flask_app.route('/api/student/login', methods=['POST'])
async def student_login():
    """
        Xử lý đăng nhập cho sinh viên.
        Nhận mã số sinh viên và mật khẩu từ request body.
        Trả về token JWT nếu đăng nhập thành công.
    """

    # Lấy dữ liệu từ request body
    data = request.get_json()

    # Kiểm tra xem dữ liệu có đầy đủ không
    if not data or not data.get('student_id') or not data.get('password'):
        return jsonify({'message': 'Thiếu mã số hoặc mật khẩu'}), 400
    
    student_id = data.get('student_id')
    password = data.get('password')

    # Tìm sinh viên trong cơ sở dữ liệu
    student = await prisma.student.find_first(
        where={
            'studentCode': student_id,
        }
    )

    # Kiểm tra xem sinh viên có tồn tại và mật khẩu có đúng không
    if not student or student.password != password:
        return jsonify({'message': 'Mã số hoặc mật khẩu không đúng'}), 401
    
    # Sinh viên đã đăng nhập thành công, tạo token JWT
    token_payload = {
        'student_id': student.id,
        'exp': datetime.utcnow() + timedelta(days=1)
    }

    # Mã hóa token với secret key
    token = jwt.encode(token_payload, flask_app.config['SECRET_KEY'], algorithm='HS256')

    # Trả về token và thông tin sinh viên
    return jsonify({
        'message': 'Đăng nhập thành công',
        'token': token,
        'student_id': student.id,
        'name': student.name
    })

# API xác thực truy cập của sinh viên
@flask_app.route('/api/student/verify-access-flexible', methods=['POST'])
@token_required
async def verify_student_access_flexible():
    """
        Xử lý xác thực truy cập cho sinh viên.
        Kiểm tra IP và GPS một cách linh hoạt.
        Truy cập được cấp nếu IP đúng (wifi trường) HOẶC GPS đúng (trong phạm vi).
    """

    # Lấy tọa độ từ request
    data = request.get_json()
    student_lat = data.get('latitude')
    student_lon = data.get('longitude')
    student_ip = data.get('ip_address')

    # Mặc định các trạng thái là không hợp lệ
    is_ip_ok = False
    is_gps_ok = False
    
    # 1. Kiểm tra IP
    if check_ip_allowed(student_ip):

        is_ip_ok = True

    # 2. Kiểm tra GPS
    if student_lat is not None and student_lon is not None:
        CAMPUS_LAT = float(os.environ.get('CAMPUS_LAT', 10.729256761720086799853))
        CAMPUS_LON = float(os.environ.get('CAMPUS_LON', 106.70934363989961))
        # CAMPUS_LAT = float(os.environ.get('CAMPUS_LAT', 10.799853))
        # CAMPUS_LON = float(os.environ.get('CAMPUS_LON', 106.654474))
        MAX_DISTANCE_METERS = 500  # Bán kính 500m

        distance_m = haversine_distance(student_lon, student_lat, CAMPUS_LON, CAMPUS_LAT)
        if (distance_m) <= MAX_DISTANCE_METERS:
            is_gps_ok = True
    
    print(f"INFO:     Kết quả kiểm tra IP: {is_ip_ok}, GPS: {is_gps_ok}")
    # 3. Quyết định cấp quyền truy cập
    if is_ip_ok and is_gps_ok:
        return jsonify({
            "accessGranted": True,
            "reason": "Xác thực thành công."
        }), 200
    else:
        # Nếu cả hai đều thất bại, trả về lỗi
        return jsonify({
            "accessGranted": False,
            "reason": "Truy cập bị từ chối. Vui lòng kết nối vào Wi-Fi của trường hoặc di chuyển vào trong khuôn viên trường."
        }), 403        

# API đăng nhập cho giáo viên
@flask_app.route('/api/teacher/login', methods=['POST'])
async def teacher_login():
    """
        Xử lý đăng nhập cho giáo viên.
        Nhận mã số giáo viên và mật khẩu từ request body.
        Trả về token JWT nếu đăng nhập thành công.
    """

    # Lấy dữ liệu từ request body
    data = request.get_json()

    # Kiểm tra xem dữ liệu có đầy đủ không
    if not data or not data.get('teacher_id') or not data.get('password'):
        return jsonify({'message': 'Thiếu mã số hoặc mật khẩu'}), 400
    
    teacher_id = data.get('teacher_id')
    password = data.get('password')

    # Tìm giáo viên trong cơ sở dữ liệu
    teacher = await prisma.teacher.find_first(
        where={
            'teacherCode': teacher_id,
        }
    )

    # Kiểm tra xem giáo viên có tồn tại và mật khẩu có đúng không
    if not teacher or teacher.password != password:
        return jsonify({'message': 'Mã số hoặc mật khẩu không đúng'}), 401

    # Giáo viên đã đăng nhập thành công, tạo token JWT
    token_payload = {
        'teacher_id': teacher.id,
        'exp': datetime.utcnow() + timedelta(days=1)
    }

    # Mã hóa token với secret key
    token = jwt.encode(token_payload, flask_app.config['SECRET_KEY'], algorithm='HS256')
    
    # Trả về token và thông tin giáo viên
    return jsonify({
        'message': 'Đăng nhập thành công',
        'token': token,
        'teacher_id': teacher.id,
        'name': teacher.name
    })

# API lấy thông tin bảng điều khiển của sinh viên
@flask_app.route('/api/student/dashboard', methods=['GET'])
@token_required
async def student_dashboard():
    try:
        current_student = g.current_student # Lấy thông tin sinh viên từ g object

        # Lấy ngày hôm nay và thứ trong tuần
        today_weekday = datetime.today().weekday()

        # Lấy các lớp học có lịch trong hôm nay
        todays_course_classes = await prisma.courseclass.find_many(
            where={
                # Lớp có sinh viên này ghi danh
                'enrollments': {
                    'some': {
                        'studentId': current_student.id
                    }
                },
                # Và có lịch học vào ngày hôm nay
                'schedules': {
                    'some': {
                        'dayOfWeek': today_weekday
                    }
                }
            },
            include={
                'teacher': True,
                'course': True,
                'schedules': {
                    'where': {
                        'dayOfWeek': today_weekday
                    }
                },
                'attendanceSessions': {
                    'where': { 'isOpen': True },
                    'include': {
                        # Lấy các bản ghi điểm danh của sinh viên này trong phiên điểm danh
                        'records': {
                            'where': { 'studentId': current_student.id }
                        }
                    }
                }
            }
        )

        # Tạo danh sách lịch học cho ngày hôm nay
        # Mỗi lớp học có thể có nhiều lịch học trong ngày, nên cần lặp qua từng lớp và từng lịch học
        schedule_today = []
        for cls in todays_course_classes:
            for schedule in cls.schedules:
                open_session = next((s for s in cls.attendanceSessions if s.isOpen), None)
                
                is_session_open = open_session is not None
                student_status = 'N/A'
                
                # Nếu có phiên điểm danh mở, lấy trạng thái của sinh viên
                if open_session and open_session.records:
                    student_status = open_session.records[0].status

                schedule_today.append({
                    'class_id': cls.id,
                    'subject_name': cls.course.courseName,
                    'teacher_name': cls.teacher.name,
                    'start_time': schedule.startTime,
                    'end_time': schedule.endTime,
                    'room': schedule.room,
                    'is_session_open': is_session_open,
                    'student_status': student_status
                })

        # Sắp xếp lịch học theo thời gian bắt đầu
        schedule_today.sort(key=lambda x: x['start_time'])

        # Lấy thống kê điểm danh của sinh viên trong ngày hôm nay
        attendance_stats_raw = await prisma.attendancerecord.group_by(
            by=['status'],
            where={'studentId': current_student.id},
            count={'status': True} 
        )
        
        # Tạo tóm tắt điểm danh
        attendance_summary = {
            'PRESENT': 0,
            'ABSENT': 0,
            'UNMARKED': 0
        }

        # Chỉ lấy các trạng thái có trong thống kê
        for stat in attendance_stats_raw:
            if stat['status'] in attendance_summary:
                # Kết quả trả về vẫn nằm trong key `_count`
                attendance_summary[stat['status']] = stat['_count']['status']

        # Tạo dữ liệu trả về cho bảng điều khiển
        dashboard_data = {
            'student_info': {
                'id': current_student.id,
                'student_id': current_student.studentCode,
                'name': current_student.name
            },
            'schedule_today': schedule_today,
            'attendance_summary': attendance_summary
        }

        return jsonify(dashboard_data)

    except Exception as e:
        # In lỗi ra console để debug
        print(f"Lỗi server: {e}")
        return jsonify({'message': 'Lỗi server nội bộ', 'error': str(e)}), 500

# API lấy lịch học của sinh viên trong ngày hôm nay
@flask_app.route('/api/student/schedule/today', methods=['GET'])
@token_required
async def student_schedule():
    try:
        current_student = g.current_student # Lấy thông tin sinh viên từ g object

        # Lấy ngày hôm nay và thứ trong tuần
        today_weekday = datetime.today().weekday()

        # Lấy các lớp học có lịch trong hôm nay
        todays_course_classes = await prisma.courseclass.find_many(
            where={
                # Lớp có sinh viên này ghi danh
                'enrollments': {
                    'some': {
                        'studentId': current_student.id
                    }
                },
                # Và có lịch học vào ngày hôm nay
                'schedules': {
                    'some': {
                        'dayOfWeek': today_weekday
                    }
                }
            },
            include={
                'teacher': True,
                'course': True,
                'schedules': {
                    'where': {
                        'dayOfWeek': today_weekday
                    }
                },
                'attendanceRecords': {
                    'where': {
                        'studentId': current_student.id
                    }
                }
            },
        )

        # Tạo danh sách lịch học cho ngày hôm nay
        schedule_today = []
        for cls in todays_course_classes:
            for schedule in cls.schedules:
                schedule_today.append({
                    'class_id': cls.id,
                    'class_code': cls.classCode,
                    'subject_name': cls.course.courseName,
                    'teacher_name': cls.teacher.name,
                    'start_time': schedule.startTime,
                    'end_time': schedule.endTime,
                    'room': schedule.room,
                })
        
        # Sắp xếp lịch học theo thời gian bắt đầu
        schedule_today.sort(key=lambda x: x['start_time'])

        return jsonify(schedule_today)
    
    except Exception as e:
        print(f"Lỗi server: {e}")
        return jsonify({'message': 'Lỗi truy xuất thông tin sinh viên'}), 500

# API lấy lịch học của sinh viên trong tuần
@flask_app.route('/api/student/schedule/week', methods=['GET'])
@token_required
async def student_week_schedule():
    try:
        current_student = g.current_student # Lấy thông tin sinh viên từ g object

        # 1. Lấy tất cả lịch học của sinh viên
        all_schedules = await prisma.classschedule.find_many(
            where={
                # Lớp học có sinh viên này ghi danh
                'course_class': {
                    'is': {
                        'enrollments': {
                            'some': {
                                'studentId': current_student.id
                            }
                        }
                    }
                }
            },
            include={
                'course_class': {
                    'include': {
                        'teacher': True,
                        'course': True
                    }
                }
            }
        )

        # 2. Nhóm lịch học theo ngày
        days_map = {
            0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
            4: "Friday", 5: "Saturday", 6: "Sunday"
        }
        
        # Khởi tạo dictionary để lưu lịch học theo ngày
        schedule_by_day = {day: [] for day in days_map.values()}
        for schedule in all_schedules:
            day_name = days_map.get(schedule.dayOfWeek)
            if day_name:
                cls = schedule.course_class
                schedule_by_day[day_name].append({
                    'class_id': cls.id,
                    'class_code': cls.classCode,
                    'subject_name': cls.course.courseName,
                    'teacher_name': cls.teacher.name,
                    'start_time': schedule.startTime,
                    'end_time': schedule.endTime,
                    'room': schedule.room,
                })

        # Sắp xếp các lớp trong mỗi ngày theo thời gian bắt đầu
        for day in schedule_by_day:
            schedule_by_day[day].sort(key=lambda x: x['start_time'])
            
        return jsonify(schedule_by_day)

    except Exception as e:
        print(f"Error in student_week_schedule: {e}")
        return jsonify({'message': 'Lỗi server nội bộ', 'error': str(e)}), 500

# API lấy lịch sử điểm danh của sinh viên
@flask_app.route('/api/student/attendance/history', methods=['GET'])
@token_required
async def student_attendance_history():
    try:
        current_student = g.current_student # Lấy thông tin sinh viên từ g object

        # Lấy tất cả bản ghi điểm danh của sinh viên này
        attendance_records = await prisma.attendancerecord.find_many(
            # Chỉ lấy các bản ghi của sinh viên hiện tại
            where={ 'studentId': current_student.id },
            include={
                'session': {
                    'include': {
                        'course_class': {
                            'include': {
                                'course': True
                            }
                        }
                    }
                }
            }
        )

        # Chỉ lấy các bản ghi đã điểm danh (PRESENT) hoặc chưa điểm danh (UNMARKED)
        history = []
        for record in attendance_records:
            history.append({
                'session_date_obj': record.session.sessionDate,
                'date': record.session.sessionDate.strftime('%Y-%m-%d'),
                'subject_name': record.session.course_class.course.courseName,
                'status': record.status,
                'check_in_time': record.checkInTime.strftime('%H:%M:%S') if record.checkInTime else None,
                'notes': record.notes
            })
        
        # Sắp xếp lịch sử theo ngày điểm danh, mới nhất trước
        history.sort(key=lambda x: x['session_date_obj'], reverse=True)

        # Chuyển đổi ngày thành chuỗi và xóa trường session_date_obj
        for item in history:
            del item['session_date_obj']

        return jsonify(history)
    except Exception as e:
        print(f"Lỗi server: {e}")
        return jsonify({'message': 'Lỗi truy xuất lịch sử điểm danh'}), 500

# API điểm danh cho sinh viên
@flask_app.route('/api/student/attendance/check-in', methods=['POST'])
@token_required
async def student_check_in():
    try:
        current_student = g.current_student # Lấy thông tin sinh viên từ g object

        print("INFO:     Bắt đầu xử lý điểm danh cho sinh viên...")
        face_image = None
        class_id = None

        # Kiểm tra xem yêu cầu có phải là multipart/form-data hay không
        if 'face_image' in request.files:
            print("INFO:     Đang xử lý yêu cầu multipart/form-data.")
            class_id = request.form.get('class_id')
            image_file = request.files['face_image']
            
            # Đọc file và mã hóa sang base64 để gửi đi
            image_bytes = image_file.read()
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            face_image = f"data:{image_file.mimetype};base64,{encoded_image}"
        
        # Giữ lại logic xử lý JSON để dự phòng
        elif request.is_json:
            print("INFO:     Đang xử lý yêu cầu application/json.")
            data = request.get_json()
            class_id = data.get('class_id')
            face_image = data.get('image') # Giả định key là 'image' cho JSON
        
        else:
            return jsonify({'message': 'Định dạng yêu cầu không được hỗ trợ. Cần multipart/form-data hoặc application/json.'}), 415

        print(f"Received class_id: {class_id}")

        # Kiểm tra xem class_id và face_image có tồn tại không
        if not class_id:
            return jsonify({'message': 'Thiếu class_id trong yêu cầu'}), 400
        if not face_image:
            return jsonify({'message': 'Thiếu dữ liệu ảnh (face_image) trong yêu cầu'}), 400

        # 1. Kiểm tra xem ảnh có phải là khuôn mặt thật không
        if not is_real_face(face_image):
            return jsonify({'message': 'Phát hiện khuôn mặt không hợp lệ'}), 400
        
        # 2. Nhận diện khuôn mặt để lấy MSSV
        student_code = recognize_face(face_image, current_student.studentCode)
        if not student_code:
            return jsonify({'message': 'Khuôn mặt không khớp'}), 400

        active_session = await prisma.attendancesession.find_first(
            where={
                'classId': class_id,
                'isOpen': True
            }
        )

        # Kiểm tra xem có phiên điểm danh nào đang mở cho lớp này không
        if not active_session:
            return jsonify({'message': 'Không có phiên điểm danh nào đang mở cho lớp này'}), 404

        # 3. Kiểm tra xem sinh viên đã điểm danh trong phiên này chưa
        existing_record = await prisma.attendancerecord.find_unique(
            where={
                'sessionId_studentId': {
                    'sessionId': active_session.id,
                    'studentId': current_student.id
                }
            }
        )

        # Nếu đã điểm danh với trạng thái 'PRESENT', trả về thông báo lỗi
        if existing_record and existing_record.status == 'PRESENT':
            return jsonify({'message': 'Bạn đã điểm danh rồi'}), 409 # 409 Conflict

        # Nếu đã điểm danh nhưng không phải 'PRESENT', cập nhật trạng thái
        if existing_record and existing_record.status == 'PRESENT':
            return jsonify({'message': 'Bạn đã điểm danh rồi'}), 409

        # Nếu chưa điểm danh, tạo bản ghi mới hoặc cập nhật bản ghi cũ
        if existing_record:
            await prisma.attendancerecord.update(
                where={'id': existing_record.id},
                data={'status': 'PRESENT', 'checkInTime': datetime.utcnow()}
            )
        else:
            await prisma.attendancerecord.create(
                data={
                    'sessionId': active_session.id,
                    'studentId': current_student.id,
                    'status': 'PRESENT',
                    'checkInTime': datetime.utcnow()
                }
            )

        # Lấy thông tin lớp để tìm teacherId
        course_class = await prisma.courseclass.find_unique(
            where={'id': class_id},
            include={'teacher': True, 'course': True}
        )

        # Nếu tìm thấy lớp học và có giáo viên, gửi thông báo
        if course_class and course_class.teacherId:
            teacher_id = course_class.teacherId
            # Tạo payload thông báo
            notification_payload = {
                "type": "new_check_in",
                "student_name": current_student.name,
                "class_code": course_class.classCode,
                "subject_name": course_class.course.courseName,
                "timestamp": datetime.utcnow().isoformat()
            }
            # Gửi thông báo qua manager
            await manager.send_personal_message(notification_payload, teacher_id)

        return jsonify({'message': 'Điểm danh thành công'})

    except Exception as e:
        print(f"Error in student_check_in: {e}")
        return jsonify({'message': 'Lỗi server nội bộ', 'error': str(e)}), 500

# API lấy thông tin bảng điều khiển của giáo viên
@flask_app.route('/api/teacher/dashboard', methods=['GET'])
@teacher_token_required
async def teacher_dashboard():
    try:
        current_teacher = g.current_teacher # Lấy thông tin giáo viên từ g object

        # Lấy ngày hôm nay và thứ trong tuần
        today_weekday = datetime.today().weekday()

        # lấy tất cả các lớp học của giáo viên này
        all_teacher_classes = await prisma.courseclass.find_many(
            where={'teacherId': current_teacher.id},
            include={'schedules': True}
        )

        # Lấy tổng số lớp học
        total_classes_count = len(all_teacher_classes)

        # 1. Lấy tất cả các lớp học của giáo viên này
        todays_classes_count = 0
        for cls in all_teacher_classes:
            # Kiểm tra xem lớp có lịch dạy hôm nay không
            if any(schedule.dayOfWeek == today_weekday for schedule in cls.schedules):
                todays_classes_count += 1
        
        # 2. Lấy tất cả các phiên điểm danh của giáo viên này
        attendance_stats = await prisma.attendancerecord.group_by(
            by=['status'],
            where={
                'session': {
                    'is': {
                        'course_class': {
                            'is': {
                                'teacherId': current_teacher.id
                            }
                        }
                    }
                },
                'status': {
                    'in': ['PRESENT', 'ABSENT']
                }
            },
            count={'_all': True}
        )

        present_count = 0
        absent_count = 0

        # 3. Tính tổng số buổi điểm danh
        for stat in attendance_stats:
            if stat['status'] == 'PRESENT':
                present_count = stat['_count']['_all']
            elif stat['status'] == 'ABSENT':
                absent_count = stat['_count']['_all']
        
        total_recorded_sessions = present_count + absent_count
        
        # Tính tỉ lệ, tránh chia cho 0
        overall_attendance_rate = (present_count / total_recorded_sessions * 100) if total_recorded_sessions > 0 else 0

        # 4. Xây dựng response
        dashboard_data = {
            'teacher_info': {
                'id': current_teacher.id,
                'teacher_id': current_teacher.teacherCode,
                'name': current_teacher.name
            },
            'summary': {
                'todays_classes_count': todays_classes_count,
                'total_classes_count': total_classes_count,
                'overall_attendance_rate': round(overall_attendance_rate, 2) # Làm tròn 2 chữ số
            }
        }

        return jsonify(dashboard_data)

    except Exception as e:
        print(f"Error in teacher_dashboard: {e}")
        return jsonify({'message': 'Lỗi server nội bộ', 'error': str(e)}), 500

# API lấy danh sách lớp học của giáo viên
@flask_app.route('/api/teacher/classes', methods=['GET'])
@teacher_token_required
async def teacher_classes():
    try:
        current_teacher = g.current_teacher # Lấy thông tin giáo viên từ g object

        # 1. Lấy tất cả các lớp học của giáo viên này
        classes = await prisma.courseclass.find_many(
            where={'teacherId': current_teacher.id},
            include={
                'course': True,
                'schedules': True,
                'enrollments': True # Lấy danh sách enrollments để đếm
            }
        )

        # 2. Định dạng dữ liệu để trả về
        classes_data = []
        for cls in classes:
            # Định dạng lại lịch học để hiển thị
            schedule_list = []
            if cls.schedules:
                # Sắp xếp lịch theo thứ tự ngày trong tuần
                sorted_schedules = sorted(cls.schedules, key=lambda s: s.dayOfWeek)
                days_map = {0: "Thứ hai", 1: "Thứ ba", 2: "Thứ tư", 3: "Thứ năm", 4: "Thứ sáu", 5: "Thứ bảy", 6: "Chủ nhật  "}
                for schedule in sorted_schedules:
                    day_name = days_map.get(schedule.dayOfWeek, "N/A")
                    schedule_list.append(f"{day_name} ({schedule.startTime}-{schedule.endTime})")
            
            classes_data.append({
                'class_id': cls.id,
                'class_code': cls.classCode,
                'subject_name': cls.course.courseName,
                'student_count': len(cls.enrollments), # Đếm số lượng enrollments
                'schedule_display': ", ".join(schedule_list),
                'start_time': cls.schedules[0].startTime if cls.schedules else None,
                'end_time': cls.schedules[0].endTime if cls.schedules else None,
                'day': cls.schedules[0].dayOfWeek if cls.schedules else None,
            })
            
        return jsonify(classes_data)
    
    except Exception as e:
        print(f"Error in teacher_classes: {e}")
        return jsonify({'message': 'Lỗi server nội bộ', 'error': str(e)}), 500

# API lấy thông tin chi tiết lớp học của giáo viên
@flask_app.route('/api/teacher/classes/<string:class_id>', methods=['GET'])
@teacher_token_required
async def teacher_class_details(class_id):
    try:
        current_teacher = g.current_teacher # Lấy thông tin giáo viên từ g object

        # 1. Tìm lớp học và đảm bảo giáo viên này có quyền truy cập
        cls = await prisma.courseclass.find_first(
            where={
                'id': class_id,
                'teacherId': current_teacher.id
            },
            include={
                'course': True,
                'schedules': True,
                'enrollments': {
                    'include': {
                        'student': True
                    }
                }
            }
        )

        if not cls:
            return jsonify({'message': 'Không tìm thấy lớp học hoặc bạn không có quyền truy cập'}), 404

        # 2. Lấy thông tin điểm danh gần nhất
        student_count = len(cls.enrollments)
        
        # Tìm phiên điểm danh gần nhất
        all_sessions = await prisma.attendancesession.find_many(
            where={'classId': class_id}
        )
        
        latest_session = None
        if all_sessions:
            all_sessions.sort(key=lambda s: s.sessionDate, reverse=True)
            latest_session = all_sessions[0]

        latest_attendance_summary = "Chưa có phiên điểm danh nào"
        attendance_status = "Chưa bắt đầu"

        if latest_session:
            present_count = await prisma.attendancerecord.count(
                where={
                    'sessionId': latest_session.id,
                    'status': 'PRESENT'
                }
            )
            latest_attendance_summary = f"{present_count}/{student_count}"
            attendance_status = "Đang mở" if latest_session.isOpen else "Đã đóng"

        # 3. Lấy và xử lý số buổi vắng
        absent_records = await prisma.attendancerecord.find_many(
            where={
                'status': 'ABSENT',
                'session': {
                    'is': {
                        'classId': class_id
                    }
                }
            }
        )
        absence_counts = {}
        for record in absent_records:
            student_id = record.studentId
            absence_counts[student_id] = absence_counts.get(student_id, 0) + 1

        # 4. Định dạng lịch học
        schedule_list = []
        if cls.schedules:
            sorted_schedules = sorted(cls.schedules, key=lambda s: s.dayOfWeek)
            days_map = {0: "Thứ hai", 1: "Thứ ba", 2: "Thứ tư", 3: "Thứ năm", 4: "Thứ sáu", 5: "Thứ bảy", 6: "Chủ nhật"}
            for schedule in sorted_schedules:
                day_name = days_map.get(schedule.dayOfWeek, "N/A")
                schedule_list.append({
                    'day': day_name,
                    'start_time': schedule.startTime,
                    'end_time': schedule.endTime,
                    'time': f"{schedule.startTime} - {schedule.endTime}",
                    'room': schedule.room
                })

        # 5. Định dạng danh sách sinh viên và thêm số buổi vắng
        student_list = []
        if cls.enrollments:
            for enrollment in cls.enrollments:
                student_id = enrollment.student.id
                student_list.append({
                    'student_db_id': student_id,
                    'student_code': enrollment.student.studentCode,
                    'name': enrollment.student.name,
                    'absent_count': absence_counts.get(student_id, 0)
                })
        
        # 6. Xây dựng response cuối cùng
        class_details = {
            'class_id': cls.id,
            'class_code': cls.classCode,
            'subject_name': cls.course.courseName,
            'semester': cls.semester,
            'student_count': student_count,
            'latest_attendance_summary': latest_attendance_summary,
            'attendance_status': attendance_status,
            'schedules': schedule_list,
            'students': student_list,
            'start_time': cls.schedules[0].startTime if cls.schedules else None,
            'end_time': cls.schedules[0].endTime if cls.schedules else None,
            'day': cls.schedules[0].dayOfWeek if cls.schedules else None
        }

        return jsonify(class_details)

    except Exception as e:
        print(f"Error in teacher_class_details: {e}")
        return jsonify({'message': 'Lỗi server nội bộ', 'error': str(e)}), 500
  
# API lấy danh sách sinh viên trong lớp học của giáo viên
@flask_app.route('/api/teacher/classes/<string:class_id>/student/attendance', methods=['GET'])
@teacher_token_required
async def teacher_class_student_attendance(class_id):
    try:
        current_teacher = g.current_teacher # Lấy thông tin giáo viên từ g object

        # 1. Xác thực lớp học thuộc về giáo viên
        cls = await prisma.courseclass.find_first(
            where={
                'id': class_id,
                'teacherId': current_teacher.id
            },
            include={
                'enrollments': {
                    'include': {
                        'student': True
                    }
                }
            }
        )

        # Nếu không tìm thấy lớp học hoặc giáo viên không có quyền truy cập
        if not cls:
            return jsonify({'message': 'Không tìm thấy lớp học hoặc bạn không có quyền truy cập'}), 404

        # 2. Tìm phiên điểm danh gần nhất
        all_sessions = await prisma.attendancesession.find_many(where={'classId': class_id})
        latest_session = None
        if all_sessions:
            all_sessions.sort(key=lambda s: s.sessionDate, reverse=True)
            latest_session = all_sessions[0]

        # 3. Lấy trạng thái điểm danh nếu có phiên
        attendance_status_map = {}
        if latest_session:
            records = await prisma.attendancerecord.find_many(
                where={'sessionId': latest_session.id}
            )
            for record in records:
                attendance_status_map[record.studentId] = record.status

        # 4. Xây dựng danh sách sinh viên trả về
        student_list = []
        for enrollment in cls.enrollments:
            student = enrollment.student
            student_list.append({
                'student_db_id': student.id,
                'student_code': student.studentCode,
                'name': student.name,
                # Lấy trạng thái từ map, mặc định là UNMARKED nếu không có
                'status': attendance_status_map.get(student.id, 'UNMARKED')
            })

        return jsonify(student_list)

    except Exception as e:
        print(f"Error in teacher_class_student_attendance: {e}")
        return jsonify({'message': 'Lỗi server nội bộ', 'error': str(e)}), 500

# API bắt đầu và dừng phiên điểm danh của giáo viên
@flask_app.route('/api/teacher/classes/<string:class_id>/attendance/start', methods=['POST'])
@teacher_token_required
async def teacher_class_attendance_start(class_id):
    try:
        current_teacher = g.current_teacher # Lấy thông tin giáo viên từ g object

        # 1. Xác thực lớp học thuộc về giáo viên
        cls = await prisma.courseclass.find_first(
            where={'id': class_id, 'teacherId': current_teacher.id},
            include={'enrollments': {'include': {'student': True}}}
        )

        # Nếu không tìm thấy lớp học hoặc giáo viên không có quyền truy cập
        if not cls:
            return jsonify({'message': 'Không tìm thấy lớp học hoặc bạn không có quyền truy cập'}), 404

        # 2. Kiểm tra xem đã có phiên điểm danh nào đang mở không
        # Nếu đã có phiên điểm danh đang mở, trả về lỗi
        existing_open_session = await prisma.attendancesession.find_first(where={'classId': class_id, 'isOpen': True})
        if existing_open_session:
            return jsonify({'message': 'Đã có một phiên điểm danh đang mở'}), 409

        # Khi GV bắt đầu thủ công -> hủy tác vụ tự động đóng để tránh xung đột.
        try:
            today_weekday = datetime.today().weekday()
            schedule_entry = await prisma.classschedule.find_first(where={'classId': class_id, 'dayOfWeek': today_weekday})
            if schedule_entry:
                auto_close_job_id = f"auto_close_{schedule_entry.id}"
                if scheduler.get_job(auto_close_job_id):
                    scheduler.remove_job(auto_close_job_id)
                    print(f"LỊCH TRÌNH: Đã xóa tác vụ theo lịch trình '{auto_close_job_id}' do bắt đầu thủ công.")
        except Exception as e:
            print(f"LỊCH TRÌNH: CẢNH BÁO - Không thể xóa tác vụ tự động trong khi bắt đầu thủ công. Lý do: {e}")

        # 3. Tạo phiên điểm danh mới
        new_session = await prisma.attendancesession.create(
            data={'classId': class_id, 'sessionDate': datetime.utcnow(), 'isOpen': True, 'openedAt': datetime.utcnow()}
        )

        # 4. Tạo bản ghi điểm danh cho tất cả sinh viên trong lớp
        # Chỉ tạo bản ghi cho những sinh viên đã ghi danh trong lớp này
        if cls.enrollments:
            records_to_create = [{'sessionId': new_session.id, 'studentId': enrollment.student.id, 'status': 'UNMARKED'} for enrollment in cls.enrollments]
            await prisma.attendancerecord.create_many(data=records_to_create)

        return jsonify({'message': 'Bắt đầu phiên điểm danh thành công', 'session_id': new_session.id}), 201

    except Exception as e:
        print(f"Error in teacher_class_attendance_start: {e}")
        return jsonify({'message': 'Lỗi server nội bộ', 'error': str(e)}), 500

# API dừng phiên điểm danh của giáo viên
@flask_app.route('/api/teacher/classes/<string:class_id>/attendance/stop', methods=['POST'])
@teacher_token_required
async def teacher_class_attendance_stop(class_id):
    try:
        current_teacher = g.current_teacher # Lấy thông tin giáo viên từ g object

        # 1. Xác thực lớp học thuộc về giáo viên
        cls = await prisma.courseclass.find_first(where={'id': class_id, 'teacherId': current_teacher.id})

        # Nếu không tìm thấy lớp học hoặc giáo viên không có quyền truy cập
        if not cls:
            return jsonify({'message': 'Không tìm thấy lớp học hoặc bạn không có quyền truy cập'}), 404

        # 2. Kiểm tra xem có phiên điểm danh nào đang mở không
        # Nếu không có phiên điểm danh đang mở, trả về lỗi
        open_session = await prisma.attendancesession.find_first(where={'classId': class_id, 'isOpen': True})
        if not open_session:
            return jsonify({'message': 'Không có phiên điểm danh nào đang mở để dừng'}), 404

        # Xoá tác vụ tự động đóng nếu có
        try:
            today_weekday = datetime.today().weekday() 
            schedule_entry = await prisma.classschedule.find_first(where={'classId': class_id, 'dayOfWeek': today_weekday})
            if schedule_entry:
                auto_close_job_id = f"auto_close_{schedule_entry.id}"
                if scheduler.get_job(auto_close_job_id):
                    scheduler.remove_job(auto_close_job_id)
                    print(f"LỊCH TRÌNH: Đã xóa tác vụ theo lịch trình '{auto_close_job_id}' do dừng thủ công.")
        except Exception as e:
            print(f"LỊCH TRÌNH: CẢNH BÁO - Không thể xóa tác vụ tự động trong khi dừng thủ công. Lý do: {e}")

        # 3. Cập nhật phiên điểm danh là đã đóng và cập nhật thời gian đóng
        await prisma.attendancesession.update(where={'id': open_session.id}, data={'isOpen': False, 'closedAt': datetime.utcnow()})
        await prisma.attendancerecord.update_many(where={'sessionId': open_session.id, 'status': 'UNMARKED'}, data={'status': 'ABSENT'})

        return jsonify({'message': 'Đã dừng phiên điểm danh thành công'})

    except Exception as e:
        print(f"Error in teacher_class_attendance_stop: {e}")
        return jsonify({'message': 'Lỗi server nội bộ', 'error': str(e)}), 500
    
# API lấy thông tin chi tiết phiên điểm danh của giáo viên
@flask_app.route('/api/teacher/classes/attendance/<string:attendance_id>')
@teacher_token_required
async def teacher_class_attendance_details(attendance_id):
    try:
        current_teacher = g.current_teacher

        # 1. Tìm phiên điểm danh và bao gồm thông tin lớp học để xác thực
        session = await prisma.attendancesession.find_unique(
            where={'id': attendance_id},
            include={
                'course_class': True,
                'records': {
                    'include': {
                        'student': True
                    }
                }
            }
        )

        # 2. Kiểm tra xem phiên có tồn tại và thuộc về giáo viên không
        if not session or session.course_class.teacherId != current_teacher.id:
            return jsonify({'message': 'Không tìm thấy phiên điểm danh hoặc bạn không có quyền truy cập'}), 404

        # 3. Định dạng danh sách sinh viên và trạng thái điểm danh
        student_records = []
        if session.records:
            for record in session.records:
                student_records.append({
                    'student_db_id': record.student.id,
                    'student_code': record.student.studentCode,
                    'name': record.student.name,
                    'status': record.status,
                    'check_in_time': record.checkInTime.strftime('%H:%M:%S') if record.checkInTime else None
                })
        
        # Sắp xếp danh sách sinh viên theo mã số
        student_records.sort(key=lambda x: x['student_code'])

        # 4. Xây dựng response cuối cùng
        session_details = {
            'session_id': session.id,
            'session_date': session.sessionDate.strftime('%Y-%m-%d'),
            'is_open': session.isOpen,
            'opened_at': session.openedAt.strftime('%H:%M:%S') if session.openedAt else None,
            'closed_at': session.closedAt.strftime('%H:%M:%S') if session.closedAt else None,
            'student_records': student_records
        }

        return jsonify(session_details)

    except Exception as e:
        print(f"Error in teacher_class_attendance_details: {e}")
        return jsonify({'message': 'Lỗi server nội bộ', 'error': str(e)}), 500

# API cập nhật điểm danh thủ công của giáo viên
@flask_app.route('/api/teacher/classes/<string:class_id>/attendance/manual-update', methods=['POST'])
@teacher_token_required
async def teacher_manual_attendance_update(class_id):
    """
    API cho phép giáo viên cập nhật điểm danh thủ công cho một sinh viên.
    Thao tác này chỉ ảnh hưởng đến phiên điểm danh gần đây nhất của lớp.
    """
    try:
        current_teacher = g.current_teacher # Lấy thông tin giáo viên từ g object

        # 1. Lấy dữ liệu từ request
        data = request.get_json()
        student_id = data.get('student_id')
        new_status = data.get('status')
        notes = data.get('notes', 'Cập nhật thủ công bởi giáo viên')

        # Kiểm tra xem student_id và new_status có hợp lệ không
        if not student_id or not new_status:
            return jsonify({'message': 'Thiếu student_id hoặc status trong yêu cầu'}), 400
        
        # Kiểm tra xem new_status có hợp lệ không
        if new_status not in ['PRESENT', 'ABSENT', 'UNMARKED']:
            return jsonify({'message': 'Trạng thái không hợp lệ. Chỉ chấp nhận PRESENT, ABSENT, UNMARKED'}), 400

        # 2. Xác thực lớp học và quyền truy cập của giáo viên
        cls = await prisma.courseclass.find_first(
            where={
                'id': class_id,
                'teacherId': current_teacher.id,
                'enrollments': {
                    'some': {
                        'studentId': student_id
                    }
                }
            }
        )

        # Nếu không tìm thấy lớp học hoặc giáo viên không có quyền truy cập
        if not cls:
            return jsonify({'message': 'Lớp học không tồn tại, bạn không có quyền truy cập, hoặc sinh viên không thuộc lớp này'}), 404

        # 3. Tìm phiên điểm danh gần nhất cho lớp này
        latest_sessions = await prisma.attendancesession.find_many(
            where={'classId': class_id},
            order={'sessionDate': 'desc'},
            take=1
        )

        # Nếu không có phiên điểm danh nào, trả về lỗi
        if not latest_sessions:
            return jsonify({'message': 'Lớp này chưa có phiên điểm danh nào'}), 404
        
        latest_session = latest_sessions[0]

        # 4. Cập nhật hoặc tạo bản ghi điểm danh cho sinh viên
        updated_record = await prisma.attendancerecord.update(
            where={
                'sessionId_studentId': {
                    'sessionId': latest_session.id,
                    'studentId': student_id
                }
            },
            data={
                'status': new_status,
                'notes': notes
            }
        )

        # Nếu không tìm thấy bản ghi để cập nhật, trả về lỗi
        if not updated_record:
             return jsonify({'message': 'Không tìm thấy bản ghi điểm danh phù hợp để cập nhật'}), 404

        return jsonify({
            'message': f"Đã cập nhật điểm danh cho sinh viên thành công.",
            'updated_status': updated_record.status
        })

    except Exception as e:
        print(f"Error in teacher_manual_attendance_update: {e}")
        return jsonify({'message': 'Lỗi server nội bộ', 'error': str(e)}), 500
    
