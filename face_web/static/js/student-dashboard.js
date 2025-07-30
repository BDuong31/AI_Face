
document.addEventListener('DOMContentLoaded', () => {
    initializeAndVerifyDashboard();
});

// Hàm chính điều phối, thực hiện xác thực TRƯỚC KHI tải dữ liệu
async function initializeAndVerifyDashboard() {
    const authToken = sessionStorage.getItem('authToken');
    if (!authToken) {
        window.location.href = '/student/login';
        return;
    }

    // Lấy tham chiếu đến các element trên DOM
    const modal = document.getElementById('access-modal');
    const modalReasonEl = document.getElementById('modal-reason-text');
    const mainContentWrapper = document.getElementById('main-content-wrapper');
    const loadingIndicator = document.getElementById('loading-indicator');

    // 1. Làm mờ nội dung chính và hiển thị màn hình chờ
    mainContentWrapper.classList.add('blur-sm', 'pointer-events-none');
    loadingIndicator.style.display = 'flex';

    try {
        // 2. Lấy tọa độ GPS
        const position = await new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(resolve, reject, {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            });
        });

        const publicIP = await getPublicIP();

        // 3. Gọi API xác thực linh hoạt
        const response = await fetch('/api/student/verify-access-flexible', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`
            },
            body: JSON.stringify({
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
                ip_address: publicIP
            })
        });

        const result = await response.json();

        if (!result.accessGranted) {
            // Nếu truy cập bị từ chối, hiển thị modal lỗi
            throw new Error(result.reason);
        }

        // 4. Nếu xác thực thành công:
        // Gỡ bỏ hiệu ứng mờ và ẩn màn hình chờ
        mainContentWrapper.classList.remove('blur-sm', 'pointer-events-none');
        loadingIndicator.style.display = 'none';

        // Bắt đầu tải dữ liệu cho dashboard
        initializeDashboardFeatures(authToken);

    } catch (error) {
        // 5. Nếu có bất kỳ lỗi nào xảy ra (từ chối GPS, API trả về lỗi)
        loadingIndicator.style.display = 'none'; // Ẩn màn hình chờ
        modalReasonEl.textContent = error.message; // Hiển thị lý do lỗi
        modal.style.display = 'flex'; // Hiển thị modal lỗi
        // Nội dung chính vẫn bị mờ và không thể tương tác
    }
}

// Hàm này chứa các logic gốc của bạn, được gọi SAU KHI xác thực thành công
function initializeDashboardFeatures(authToken) {
    // Gọi các hàm tải dữ liệu
    fetchDashboardData(authToken);
    fetchWeeklySchedule(authToken);
    fetchAttendanceHistory(authToken);

    // Gán sự kiện cho nút logout
    document.getElementById('logoutButton').addEventListener('click', () => {
        sessionStorage.removeItem('authToken');
        sessionStorage.removeItem('studentInfo');
        window.location.href = '/student/login';
    });

    // Gán sự kiện cho modal thông báo thiếu dữ liệu khuôn mặt
    const noticeModal = document.getElementById('faceDataNoticeModal');
    const closeButton = document.getElementById('closeFaceNoticeModal');
    if (noticeModal && closeButton) {
        closeButton.addEventListener('click', () => {
            noticeModal.classList.add('hidden');
        });
    }
    
    // Thiết lập modal camera
    setupCameraModal(authToken);
}


// --- CÁC HÀM CÒN LẠI GIỮ NGUYÊN ---

async function checkFaceDataExists(studentId) {
    const recognitionServiceUrl = 'http://127.0.0.1:5001';
    try {
        const response = await fetch(`${recognitionServiceUrl}/api/student/exists/${studentId}`);
        if (!response.ok) {
            console.error("Lỗi khi kiểm tra dữ liệu khuôn mặt:", response.statusText);
            return { exists: true };
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Không thể kết nối đến service nhận diện:", error);
        return { exists: true };
    }
}

async function fetchWithAuth(url, token, options = {}) {
    const headers = {
        ...options.headers,
        'Authorization': `Bearer ${token}`
    };
    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
    }
    const response = await fetch(url, { ...options, headers });
    if (response.status === 401) {
        sessionStorage.clear();
        window.location.href = '/student/login';
        throw new Error('Phiên đăng nhập hết hạn.');
    }
    return response;
}

async function fetchDashboardData(token) {
    try {
        const response = await fetchWithAuth('/api/student/dashboard', token);
        const data = await response.json();

        const studentInfo = data.student_info;
        document.getElementById('studentName').textContent = studentInfo.name;
        document.getElementById('studentNameWelcome').textContent = studentInfo.name;

        const summary = data.attendance_summary;
        document.getElementById('todayClassesCount').textContent = data.schedule_today.length;
        document.getElementById('presentCount').textContent = summary.PRESENT || 0;
        document.getElementById('absentCount').textContent = summary.ABSENT || 0;
        renderDailySchedule(data.schedule_today);

        if (studentInfo && studentInfo.student_id) {
            const faceData = await checkFaceDataExists(studentInfo.student_id);
            if (faceData.exists === false) {
                const modal = document.getElementById('faceDataNoticeModal');
                if (modal) {
                    modal.classList.remove('hidden');
                }
            }
        }
    } catch (error) {
        console.error("Lỗi khi tải dữ liệu dashboard:", error);
    }
}

function renderDailySchedule(schedule) {
    const content = document.getElementById('dailyScheduleContent');
    const noClassesMsg = document.getElementById('noDailyClasses');
    content.innerHTML = '';

    if (schedule && schedule.length > 0) {
        noClassesMsg.style.display = 'none';

        schedule.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'daily-class-item p-4 rounded-lg bg-slate-50 hover:bg-slate-100 smooth-transition';

            let attendanceActionHtml = '';

            // LOGIC: Conditionally render the button or a status based on API data
            
            // Case 1: Student has already checked in successfully
            if (item.student_status === 'PRESENT') {
                attendanceActionHtml = `
                    <div class="mt-3 w-full flex items-center justify-center space-x-2 bg-green-100 text-green-700 font-semibold py-2 px-4 rounded-lg">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        <span>Đã điểm danh</span>
                    </div>`;
            } 
            // Case 2: The attendance session is open and student has not checked in
            else if (item.is_session_open) {
                attendanceActionHtml = `
                    <button data-class-id="${item.class_id}" data-class-name="${item.subject_name}" class="checkInButton mt-3 w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg flex items-center justify-center space-x-2 transition-colors">
                        <svg class="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M6.827 6.175A2.31 2.31 0 0 1 5.186 7.23c-.38.054-.757.112-1.134.174C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 0 0-1.134-.174 2.31 2.31 0 0 1-1.64-1.055l-.822-1.316a2.192 2.192 0 0 0-1.736-1.039 48.774 48.774 0 0 0-5.232 0 2.192 2.192 0 0 0-1.736 1.039l-.821 1.316Z"></path></svg>
                        <span>Điểm Danh Ngay</span>
                    </button>`;
            } 
            // Case 3: The session has not been opened by the teacher yet
            else {
                attendanceActionHtml = `
                    <div class="mt-3 w-full flex items-center justify-center space-x-2 bg-gray-200 text-gray-500 font-semibold py-2 px-4 rounded-lg cursor-not-allowed">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        <span>Chưa mở điểm danh</span>
                    </div>`;
            }

            itemDiv.innerHTML = `
                <div class="class-info">
                    <p class="font-semibold">${item.subject_name}</p>
                    <p class="text-sm text-gray-600">Thời gian: ${extractTime(item.start_time)} - ${extractTime(item.end_time)} | Phòng: ${item.room}</p>
                    <p class="text-sm text-gray-600">GV: ${item.teacher_name}</p>
                </div>
                <div class="attendance-action-container">
                    ${attendanceActionHtml}
                </div>`;
            
            content.appendChild(itemDiv);

            itemDiv.addEventListener('click', (e) => {
                if (e.target.closest('button')) return;
                itemDiv.classList.toggle('expanded');
            });
        });
        
        // Re-attach listeners to only the newly created clickable buttons
        document.querySelectorAll('.checkInButton').forEach(button => {
            button.addEventListener('click', handleCheckInButtonClick);
        });

    } else {
        noClassesMsg.textContent = "Không có lịch học nào hôm nay.";
        noClassesMsg.style.display = 'block';
    }
}

function extractTime(datetime) {
    return datetime.split(" ")[1].slice(0, 5);
}

async function fetchWeeklySchedule(token) {
    const tableBody = document.getElementById('scheduleTableBody');
    try {
        const response = await fetchWithAuth('/api/student/schedule/week', token);
        const scheduleByDay = await response.json();
        tableBody.innerHTML = '';
        const daysOrder = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
        const daysName = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"];
        let hasSchedule = false;
        daysOrder.forEach(dayName => {
            if (scheduleByDay[dayName] && scheduleByDay[dayName].length > 0) {
                hasSchedule = true;
                scheduleByDay[dayName].forEach(item => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td class="px-4 py-3 text-sm font-semibold text-gray-900">${daysName[daysOrder.indexOf(dayName)]}</td>
                        <td class="px-4 py-3 text-sm text-gray-800">${item.subject_name}</td>
                        <td class="px-4 py-3 text-sm text-gray-500">${extractTime(item.start_time)} - ${extractTime(item.end_time)}</td>
                        <td class="px-4 py-3 text-sm text-gray-500">${item.room}</td>
                        <td class="px-4 py-3 text-sm text-gray-500">${item.teacher_name}</td>
                    `;
                    tableBody.appendChild(row);
                });
            }
        });
        if (!hasSchedule) {
            tableBody.innerHTML = `<tr><td colspan="5" class="p-4 text-center text-gray-500">Không có lịch học trong tuần.</td></tr>`;
        }
    } catch (error) {
        console.error("Lỗi tải lịch học tuần:", error);
        tableBody.innerHTML = `<tr><td colspan="5" class="p-4 text-center text-red-500">Lỗi tải dữ liệu.</td></tr>`;
    }
}

async function fetchAttendanceHistory(token) {
    const tableBody = document.getElementById('attendanceTableBody');
    try {
        const response = await fetchWithAuth('/api/student/attendance/history', token);
        const history = await response.json();
        tableBody.innerHTML = '';
        if (history.length > 0) {
            history.forEach(item => {
                let statusClass = 'bg-gray-100 text-gray-800';
                let statusText = 'Chưa điểm danh';
                if (item.status === "PRESENT") {
                    statusClass = 'bg-green-100 text-green-800';
                    statusText = 'Có mặt';
                } else if (item.status === "ABSENT") {
                    statusClass = 'bg-red-100 text-red-800';
                    statusText = 'Vắng';
                }
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-800">${item.date}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-900">${item.subject_name}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${item.check_in_time || 'N/A'}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-center text-sm">
                        <span class="px-3 py-1 rounded-full ${statusClass}">${statusText}</span>
                    </td>
                `;
                tableBody.appendChild(row);
});
        } else {
            tableBody.innerHTML = `<tr><td colspan="4" class="p-4 text-center text-gray-500">Chưa có lịch sử điểm danh.</td></tr>`;
        }
    } catch (error) {
        console.error("Lỗi tải lịch sử điểm danh:", error);
        tableBody.innerHTML = `<tr><td colspan="4" class="p-4 text-center text-red-500">Lỗi tải dữ liệu.</td></tr>`;
    }
}

let videoStream = null;
let currentCheckingInClassId = null;
const cameraModal = document.getElementById('cameraModal');

function handleCheckInButtonClick(event) {
    currentCheckingInClassId = event.currentTarget.dataset.classId;
    document.getElementById('modalClassName').textContent = event.currentTarget.dataset.className;
    cameraModal.classList.add('active');
    startCamera();
}

async function submitCheckIn(classId, imageBlob, token) {
    const captureButton = document.getElementById('captureFaceButton');
    captureButton.disabled = true;
    const statusEl = document.getElementById('modalCheckInStatus');
    statusEl.textContent = "Đang xử lý...";
    statusEl.className = "text-sm h-5 text-blue-600";
    const formData = new FormData();
    formData.append('class_id', classId);
    formData.append('face_image', imageBlob, 'face.jpg');
    try {
        const response = await fetchWithAuth('/api/student/attendance/check-in', token, {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (response.ok) {
            statusEl.textContent = result.message;
            statusEl.className = "text-sm h-5 text-green-600";
            setTimeout(() => {
                cameraModal.classList.remove('active');
                stopCamera();
                fetchDashboardData(token);
            }, 1500);
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        statusEl.textContent = error.message || "Điểm danh thất bại.";
        statusEl.className = "text-sm h-5 text-red-600";
        captureButton.disabled = false;
    }
}

function setupCameraModal(token) {
    const closeButton = document.getElementById('closeModalButton');
    const captureButton = document.getElementById('captureFaceButton');
    closeButton.addEventListener('click', () => { cameraModal.classList.remove('active'); stopCamera(); });
    cameraModal.addEventListener('click', (e) => { if (e.target === cameraModal) { cameraModal.classList.remove('active'); stopCamera(); } });
    captureButton.addEventListener('click', () => {
        if (!videoStream || !currentCheckingInClassId) return;
        const video = document.getElementById('cameraFeed');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        canvas.toBlob(blob => {
            submitCheckIn(currentCheckingInClassId, blob, token);
        }, 'image/jpeg');
    });
}

async function startCamera() {
    const captureButton = document.getElementById('captureFaceButton');
    const cameraMessage = document.getElementById('cameraMessage');
    const videoFeed = document.getElementById('cameraFeed');
    captureButton.disabled = false;
    cameraMessage.textContent = "Đang yêu cầu quyền truy cập camera...";
    cameraMessage.style.display = 'block';
    videoFeed.style.display = 'none';
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
        videoFeed.srcObject = videoStream;
        videoFeed.style.display = 'block';
        cameraMessage.style.display = 'none';
    } catch (err) {
        cameraMessage.textContent = "Lỗi: Không thể truy cập camera. Vui lòng cấp quyền và thử lại.";
        console.error("Camera error:", err);
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
}

async function getPublicIP() {
  try {
    const response = await fetch('https://api.ipify.org?format=json');
    
    // Kiểm tra nếu yêu cầu không thành công
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    
    const data = await response.json();
    console.log('IP Public của bạn:', data.ip);
    
    // Ví dụ hiển thị IP lên trang web
    // document.getElementById('ip-display').textContent = data.ip;
    
    return data.ip;

  } catch (error) {
    console.error('Không thể lấy địa chỉ IP:', error);
    // document.getElementById('ip-display').textContent = 'Lỗi!';
  }
}

