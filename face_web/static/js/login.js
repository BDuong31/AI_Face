// Đặt năm hiện tại cho footer
// document.getElementById('currentYear').textContent = new Date().getFullYear();

// Lấy các phần tử từ DOM
const loginForm = document.getElementById('unifiedLoginForm');
const loginStatus = document.getElementById('loginStatus');

// Chỉ thêm trình lắng nghe sự kiện nếu biểu mẫu tồn tại
if (loginForm) {
    loginForm.addEventListener('submit', function (event) {
        event.preventDefault(); // Ngăn chặn việc gửi biểu mẫu mặc định

        // Lấy giá trị từ các trường nhập liệu
        const studentId = document.getElementById("loginIdentifier").value;
        const password = document.getElementById("password").value;

        // Kiểm tra đầu vào cơ bản
        if (!studentId || !password) {
            loginStatus.textContent = 'Vui lòng nhập đầy đủ thông tin.';
            loginStatus.className = 'mt-5 text-sm text-center h-5 text-red-600';
            return;
        }

        // Hiển thị trạng thái đang xử lý
        loginStatus.textContent = "Đang xử lý...";
        loginStatus.className = 'mt-5 text-sm text-center h-5 text-gray-600';

        // Gửi yêu cầu đăng nhập đến API
        fetch("/api/student/login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            // Dữ liệu gửi đi khớp với yêu cầu của API
            body: JSON.stringify({ student_id: studentId, password: password }),
        })
        .then(async (response) => {
            const data = await response.json();
            if (!response.ok) {
                // Ném lỗi với thông báo từ server để catch xử lý
                throw new Error(data.message || 'Lỗi mạng hoặc máy chủ.');
            }
            return data;
        })
        .then((data) => {
            // Lưu token và thông tin sinh viên nếu đăng nhập thành công
            sessionStorage.setItem("authToken", data.token);
            sessionStorage.setItem(
                "studentInfo",
                JSON.stringify({ student_id: studentId, name: data.name })
            );

            loginStatus.textContent = "Đăng nhập thành công! Đang chuyển hướng...";
            loginStatus.className = 'mt-5 text-sm text-center h-5 text-green-600';

            // Chuyển hướng sau một khoảng thời gian ngắn
            setTimeout(() => {
                // Chuyển hướng đến trang dashboard của sinh viên
                window.location.href = "/student/dashboard";
            }, 1000);
        })
        .catch((error) => {
            // Hiển thị thông báo lỗi nếu có sự cố
            loginStatus.textContent = error.message || "Thông tin đăng nhập không chính xác.";
            loginStatus.className = 'mt-5 text-sm text-center h-5 text-red-600';
        });
    });
}