document.addEventListener('DOMContentLoaded', () => {
    // document.getElementById('currentYear').textContent = new Date().getFullYear();

    const loginForm = document.getElementById('unifiedLoginForm');
    const loginStatus = document.getElementById('loginStatus');

    if (loginForm) {
        loginForm.addEventListener("submit", function (event) {
            event.preventDefault();
            
            const teacherId = document.getElementById("loginIdentifier").value;
            const password = document.getElementById("password").value;
            
            if (!teacherId || !password) {
                loginStatus.textContent = 'Vui lòng nhập đầy đủ thông tin.';
                loginStatus.className = 'mt-5 text-sm text-center h-5 text-red-600';
                return;
            }

            loginStatus.textContent = 'Đang xác thực...';
            loginStatus.className = 'mt-5 text-sm text-center h-5 text-gray-600';

            fetch("/api/teacher/login", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ teacher_id: teacherId, password: password }),
            })
            .then(async (response) => {
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.message || 'Lỗi mạng hoặc máy chủ.');
                }
                return data;
            })
            .then((data) => {
                // Lưu token và thông tin giáo viên
                sessionStorage.setItem("teacherAuthToken", data.token);
                sessionStorage.setItem(
                    "teacherInfo",
                    JSON.stringify({ teacher_id: teacherId, name: data.name })
                );

                loginStatus.textContent = "Đăng nhập thành công! Đang chuyển hướng...";
                loginStatus.className = 'mt-5 text-sm text-center h-5 text-green-600';
                
                setTimeout(() => {
                    window.location.href = "/teacher/dashboard";
                }, 1000);
            })
            .catch((error) => {
                loginStatus.textContent = error.message || "Thông tin đăng nhập không chính xác.";
                loginStatus.className = 'mt-5 text-sm text-center h-5 text-red-600';
            });
        });
    }
});