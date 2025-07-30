document.addEventListener('DOMContentLoaded', () => {

    // --- CONFIG & AUTH ---
    const API_BASE_URL = 'http://127.0.0.1:3000'; // Đảm bảo port chính xác
    const authToken = sessionStorage.getItem('teacherAuthToken');

    if (!authToken) {
        alert('Bạn chưa đăng nhập. Sẽ chuyển hướng đến trang đăng nhập.');
        window.location.href = '/teacher/login'; 
        return;
    }

    function extractTime(datetime) {
        return datetime.split(" ")[1].slice(0, 5);
    }
    // --- UI ELEMENTS ---
    const elements = {
        lecturerName: document.getElementById('lecturerName'),
        lecturerNameWelcome: document.getElementById('lecturerNameWelcome'),
        todayClassesCountGV: document.getElementById('todayClassesCountGV'),
        totalManagedClasses: document.getElementById('totalManagedClasses'),
        overallAttendanceRate: document.getElementById('overallAttendanceRate'),
        currentClassDetails: document.getElementById('currentClassDetails'),
        noCurrentClassMessage: document.getElementById('noCurrentClassMessage'),
        managedClassesTableBody: document.getElementById('managedClassesTableBody'),
        classDetailsModal: document.getElementById('classDetailsModal'),
        closeClassDetailsModalButton: document.getElementById('closeClassDetailsModalButton'),
        classDetailsModalTitle: document.getElementById('classDetailsModalTitle'),
        classDetailsModalBody: document.getElementById('classDetailsModalBody'),
        classDetailsModalClose: document.getElementById('classDetailsModalClose'),
        classDetailsModalToggleAttendance: document.getElementById('classDetailsModalToggleAttendance'),
        classDetailsModalViewAttendanceHistory: document.getElementById('classDetailsModalViewAttendanceHistory'),
        mobileMenuButton: document.getElementById('mobile-menu-button'),
        mobileMenu: document.getElementById('mobile-menu'),
        iconOpen: document.getElementById('icon-open'),
        iconClose: document.getElementById('icon-close'),
    };

    let currentViewingClassId = null;

    function showToast(message, duration = 5000) {
        // 1. Tạo element toast
        const toast = document.createElement('div');
        // Sửa: Dùng `transition-all` để áp dụng hiệu ứng cho cả transform và opacity
        toast.className = 'fixed bottom-5 right-5 bg-green-500 text-white py-3 px-5 rounded-lg shadow-lg transform transition-all duration-300 translate-y-20 opacity-0';
        toast.innerHTML = `
            <div class="flex items-center space-x-3">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                <span>${message}</span>
            </div>
        `;
        document.body.appendChild(toast);

        // 2. Animation hiện ra (không thay đổi)
        // Dùng setTimeout nhỏ để trình duyệt kịp render trạng thái ban đầu trước khi transition
        setTimeout(() => {
            toast.classList.remove('translate-y-20', 'opacity-0');
        }, 100);

        // 3. Tự động ẩn đi
        // Sửa: Sử dụng tham số `duration` thay vì hardcode 5000
        setTimeout(() => {
            // Sửa: Xóa class `opacity-100` không cần thiết và thêm class để ẩn đi
            toast.classList.add('opacity-0');
            // Lắng nghe sự kiện transition kết thúc thì xóa element khỏi DOM
            toast.addEventListener('transitionend', () => toast.remove(), { once: true });
        }, duration);
    }

    // --- API HELPER ---
    async function fetchWithAuth(endpoint, options = {}) {
        const headers = {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${authToken}`,
            ...options.headers,
        };
        const response = await fetch(`${API_BASE_URL}${endpoint}`, { ...options, headers });
        
        if (response.status === 401) {
            localStorage.removeItem('teacherToken');
            alert('Phiên đăng nhập hết hạn. Vui lòng đăng nhập lại.');
            window.location.href = '/teacher/login';
            throw new Error('Unauthorized');
        }
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(errorData.message || `Request failed with status ${response.status}`);
        }
        return response.json();
    }

    // --- API CALLS ---
    const api = {
        getDashboardData: () => fetchWithAuth('/api/teacher/dashboard'),
        getManagedClasses: () => fetchWithAuth('/api/teacher/classes'),
        getClassDetails: (classId) => fetchWithAuth(`/api/teacher/classes/${classId}`),
        getStudentAttendance: (classId) => fetchWithAuth(`/api/teacher/classes/${classId}/student/attendance`),
        startAttendance: (classId) => fetchWithAuth(`/api/teacher/classes/${classId}/attendance/start`, { method: 'POST' }),
        stopAttendance: (classId) => fetchWithAuth(`/api/teacher/classes/${classId}/attendance/stop`, { method: 'POST' }),
        updateManualAttendance: (classId, studentId, status) => fetchWithAuth(`/api/teacher/classes/${classId}/attendance/manual-update`, {
            method: 'POST',
            body: JSON.stringify({ student_id: studentId, status: status }),
        }),
    };

    // --- UI UPDATE FUNCTIONS ---
    function updateDashboardUI(data) {
        const { teacher_info, summary } = data;
        elements.lecturerName.textContent = teacher_info.name;
        elements.lecturerNameWelcome.textContent = teacher_info.name;
        elements.todayClassesCountGV.textContent = summary.todays_classes_count;
        elements.totalManagedClasses.textContent = summary.total_classes_count;
        elements.overallAttendanceRate.textContent = `${summary.overall_attendance_rate}%`;
    }

    function updateManagedClassesTable(classes) {
        if (!classes || classes.length === 0) {
            elements.managedClassesTableBody.innerHTML = `<tr><td colspan="5" class="text-center py-4 text-gray-500">Bạn không quản lý lớp học nào.</td></tr>`;
            return;
        }
        const dayMap = { 0: "Thứ hai", 1: "Thứ ba", 2: "Thứ tư", 3: "Thứ năm", 4: "Thứ sáu", 5: "Thứ bảy", 6: "Chủ nhật" };
        elements.managedClassesTableBody.innerHTML = classes.map((cls, index) => `
            <tr class="${index % 2 === 0 ? '' : 'bg-gray-50'}">
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-indigo-600">${cls.class_code}</td>
                <td class="px-6 py-4 text-sm text-gray-700 font-semibold table-cell-truncate" title="${cls.subject_name}">${cls.subject_name}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 hidden sm:table-cell">${cls.student_count}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 hidden md:table-cell table-cell-truncate" title="${cls.schedule_display}">${dayMap[cls.day]}, ${extractTime(cls.start_time)} - ${extractTime(cls.end_time)}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-center">
                    <button data-class-id="${cls.class_id}" class="viewClassDetailsButtonTable text-sky-600 hover:text-sky-800 smooth-transition p-1 inline-flex items-center" title="Xem chi tiết lớp học">
                        <svg class="w-5 h-5 mr-1" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" /><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" /></svg>
                        Chi tiết
                    </button>
                </td>
            </tr>
        `).join('');
        
        document.querySelectorAll('.viewClassDetailsButtonTable').forEach(button => {
            button.addEventListener('click', (e) => openClassDetailsModal(e.currentTarget.dataset.classId));
        });
    }
    
    // async function updateNextClassSection(classes) {
    //     const dayMap = { 0: "Thứ hai", 1: "Thứ ba", 2: "Thứ tư", 3: "Thứ năm", 4: "Thứ sáu", 5: "Thứ bảy", 6: "Chủ nhật" };
    //     const todayShortName = dayMap[new Date().getDay() -1];
    //     const todayClass = classes.filter(cls => cls.schedule_display.includes(todayShortName));
    //     const test = classes.filter(cls => cls.schedule_display.includes(todayShortName));
    //     console.log("test", test);
    //     console.log(todayClass);
    //     console.log(classes);
    //     console.log('Today\'s Class:', todayClass);
    //     elements.currentClassDetails.innerHTML = '';
    //     // if(todayClass) {
    //     //     const details = await api.getClassDetails(todayClass.class_id);
    //     //     console.log('Current Class Details:', details.start_time, details.end_time);
    //     //     const isAttendanceOpen = details.attendance_status === "Đang mở";
    //         //  elements.currentClassDetails.innerHTML = `
    //         //     <div class="bg-indigo-50 p-4 rounded-lg">
    //         //         <p class="text-indigo-800 font-semibold text-lg">${details.subject_name} (${details.class_code})</p>
    //         //         <p class="text-indigo-700 text-sm">Lịch trình: ${dayMap[details.day]}, ${extractTime(details.start_time)} - ${extractTime(details.end_time)}</p>
    //         //         <p class="text-indigo-700 text-sm">Sĩ số: ${details.student_count}</p>
    //         //         <div class="mt-4 flex flex-col sm:flex-row gap-3">
    //         //             <button data-class-id="${details.class_id}" id="quickToggleAttendanceButton" class="${isAttendanceOpen ? 'bg-yellow-500 hover:bg-yellow-600' : 'bg-green-500 hover:bg-green-600'} text-white font-semibold py-2 px-4 rounded-md shadow-sm smooth-transition flex items-center justify-center space-x-2 text-sm">
    //         //                 <svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15.75 15.75l-2.489-2.489m0 0a3.375 3.375 0 10-4.773-4.773 3.375 3.375 0 004.774 4.774zM21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
    //         //                 <span>${isAttendanceOpen ? 'Đóng Điểm Danh' : 'Bắt Đầu Điểm Danh'}</span>
    //         //             </button>
    //         //             <button data-class-id="${details.class_id}" class="viewClassDetailsButtonMain bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-md shadow-sm smooth-transition flex items-center justify-center space-x-2 text-sm">
    //         //                 <svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" /><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" /></svg>
    //         //                 <span>Xem Chi Tiết Lớp</span>
    //         //             </button>
    //         //         </div>
    //         //     </div>
    //         // `;
    //     //     document.querySelector('.viewClassDetailsButtonMain').addEventListener('click', (e) => openClassDetailsModal(e.currentTarget.dataset.classId));
    //     //     document.getElementById('quickToggleAttendanceButton').addEventListener('click', (e) => handleToggleAttendanceClick(e.currentTarget.dataset.classId, true));
    //     // } else {
    //     //      elements.noCurrentClassMessage.classList.remove('hidden');
    //     // }
    //         if (todayClass && todayClass.length > 0) {
    //             elements.currentClassDetails.classList.remove('hidden');
    //             elements.noCurrentClassMessage.classList.add('hidden');

    //             const fragment = document.createDocumentFragment();

    //             for (const cls of todayClass) {
    //                 const details = await api.getClassDetails(cls.class_id);
    //                 console.log('Current Class Details:', details.start_time, details.end_time);
    //                 const isAttendanceOpen = details.attendance_status === "Đang mở";
                    
    //                 const classCard = document.createElement('div');
    //                 classCard.className = 'bg-indigo-50 p-4 rounded-lg shadow-sm mb-4';

    //                 classCard.innerHTML = `
    //                     <div class="bg-indigo-50 p-4 rounded-lg">
    //                         <p class="text-indigo-800 font-semibold text-lg">${details.subject_name} (${details.class_code})</p>
    //                         <p class="text-indigo-700 text-sm">Lịch trình: ${dayMap[details.day]}, ${extractTime(details.start_time)} - ${extractTime(details.end_time)}</p>
    //                         <p class="text-indigo-700 text-sm">Sĩ số: ${details.student_count}</p>
    //                         <div class="mt-4 flex flex-col sm:flex-row gap-3">
    //                             <button data-class-id="${details.class_id}" id="quickToggleAttendanceButton" class="${isAttendanceOpen ? 'bg-yellow-500 hover:bg-yellow-600' : 'bg-green-500 hover:bg-green-600'} text-white font-semibold py-2 px-4 rounded-md shadow-sm smooth-transition flex items-center justify-center space-x-2 text-sm">
    //                                 <svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15.75 15.75l-2.489-2.489m0 0a3.375 3.375 0 10-4.773-4.773 3.375 3.375 0 004.774 4.774zM21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
    //                                 <span>${isAttendanceOpen ? 'Đóng Điểm Danh' : 'Bắt Đầu Điểm Danh'}</span>
    //                             </button>
    //                             <button data-class-id="${details.class_id}" class="viewClassDetailsButtonMain bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-md shadow-sm smooth-transition flex items-center justify-center space-x-2 text-sm">
    //                                 <svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" /><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" /></svg>
    //                                 <span>Xem Chi Tiết Lớp</span>
    //                             </button>
    //                         </div>
    //                     </div>
    //                 `;
    //                 classCard.querySelector('.quickToggleAttendanceButton').addEventListener('click', (e) => handleToggleAttendanceClick(e.currentTarget.dataset.classId, true));
    //                 classCard.querySelector('.viewClassDetailsButtonMain').addEventListener('click', (e) => openClassDetailsModal(e.currentTarget.dataset.classId));
                    
    //                 fragment.appendChild(classCard);
    //             }
    //             elements.currentClassDetails.appendChild(fragment);
    //         } else {
    //         elements.currentClassDetails.classList.add('hidden');
    //         elements.noCurrentClassMessage.classList.remove('hidden');
    //     }
    // }

    async function updateNextClassSection(classes) {
    // A map for Vietnamese day names, indexed 0 (Monday) to 6 (Sunday).
    const dayMap = { 0: "Thứ hai", 1: "Thứ ba", 2: "Thứ tư", 3: "Thứ năm", 4: "Thứ sáu", 5: "Thứ bảy", 6: "Chủ nhật" };

    // Correctly get today's index (0 for Monday, 6 for Sunday).
    // The modulo operator (%) ensures the index wraps around correctly for Sunday (getDay()=0).
    const todayIndex = (new Date().getDay() + 6) % 7;
    const todayShortName = dayMap[todayIndex];

    // Filter classes scheduled for today.
    const todayClasses = classes.filter(cls => cls.schedule_display.includes(todayShortName));

    // Clear previous content.
    elements.currentClassDetails.innerHTML = '';

    if (todayClasses && todayClasses.length > 0) {
        elements.currentClassDetails.classList.remove('hidden');
        elements.noCurrentClassMessage.classList.add('hidden');

        // Use a DocumentFragment for efficient DOM updates.
        const fragment = document.createDocumentFragment();

        for (const cls of todayClasses) {
            // Fetch details for each class.
            const details = await api.getClassDetails(cls.class_id);
            const isAttendanceOpen = details.attendance_status === "Đang mở";

            const classCard = document.createElement('div');
            // Add a margin-bottom to space out multiple cards.
            classCard.className = 'bg-indigo-50 p-4 rounded-lg shadow-sm mb-4';

            classCard.innerHTML = `
                <p class="text-indigo-800 font-semibold text-lg">${details.subject_name} (${details.class_code})</p>
                <p class="text-indigo-700 text-sm">Lịch trình: ${details.schedule_display}</p>
                <p class="text-indigo-700 text-sm">Sĩ số: ${details.student_count}</p>
                <div class="mt-4 flex flex-col sm:flex-row gap-3">
                    <button data-class-id="${details.class_id}" class="quickToggleAttendanceButton ${isAttendanceOpen ? 'bg-yellow-500 hover:bg-yellow-600' : 'bg-green-500 hover:bg-green-600'} text-white font-semibold py-2 px-4 rounded-md shadow-sm smooth-transition flex items-center justify-center space-x-2 text-sm">
                        <svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15.75 15.75l-2.489-2.489m0 0a3.375 3.375 0 10-4.773-4.773 3.375 3.375 0 004.774 4.774zM21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                        <span>${isAttendanceOpen ? 'Đóng Điểm Danh' : 'Bắt Đầu Điểm Danh'}</span>
                    </button>
                    <button data-class-id="${details.class_id}" class="viewClassDetailsButtonMain bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-md shadow-sm smooth-transition flex items-center justify-center space-x-2 text-sm">
                        <svg class="w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" /><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" /></svg>
                        <span>Xem Chi Tiết Lớp</span>
                    </button>
                </div>
            `;
            
            // Attach event listeners to the buttons inside the newly created card.
            // These selectors now correctly target classes.
            classCard.querySelector('.quickToggleAttendanceButton').addEventListener('click', (e) => handleToggleAttendanceClick(e.currentTarget.dataset.classId, true));
            classCard.querySelector('.viewClassDetailsButtonMain').addEventListener('click', (e) => openClassDetailsModal(e.currentTarget.dataset.classId));
            
            fragment.appendChild(classCard);
        }
        // Append all cards to the DOM at once.
        elements.currentClassDetails.appendChild(fragment);
    } else {
        // If no classes are found for today, show the message.
        elements.currentClassDetails.classList.add('hidden');
        elements.noCurrentClassMessage.classList.remove('hidden');
    }
}
    
    async function openClassDetailsModal(classId) {
        currentViewingClassId = classId;
        elements.classDetailsModal.classList.add('active');
        elements.classDetailsModalBody.innerHTML = '<div class="loader mx-auto"></div>';

        try {
            const classDetails = await api.getClassDetails(classId);
            console.log(classDetails);
            const studentAttendance = await api.getStudentAttendance(classId);
            populateClassDetailsModal(classDetails, studentAttendance);
        } catch (error) {
            console.error('Error loading class details:', error);
            elements.classDetailsModalBody.innerHTML = `<p class="text-red-500 text-center">Lỗi tải dữ liệu: ${error.message}</p>`;
        }
    }

    // function populateClassDetailsModal(details, studentList) {
    //     const daysName = ["Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy", "Chủ nhật"];
    //     elements.classDetailsModalTitle.textContent = `Chi Tiết Lớp: ${details.subject_name} (${details.class_code})`;
    //     const isAttendanceOpen = details.attendance_status === "Đang mở";
    //     const statusTextMap = { PRESENT: "Có mặt", ABSENT: "Vắng", UNMARKED: "Chưa điểm danh" };
    //     const statusClassMap = { PRESENT: "text-green-600", ABSENT: "text-red-600", UNMARKED: "text-gray-500" };
    //     console.log('Class Details:', details);
    //     elements.classDetailsModalBody.innerHTML = `
    //         <div>
    //             <h4 class="text-lg font-semibold text-slate-700 mb-2">Thông Tin Chung</h4>
    //             <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2 text-sm">
    //                 <p><strong class="font-medium text-gray-600">Mã lớp:</strong> ${details.class_code}</p>
    //                 <p><strong class="font-medium text-gray-600">Sĩ số:</strong> ${details.student_count} sinh viên</p>
    //                 <p><strong class="font-medium text-gray-600">Điểm danh (gần nhất):</strong> <span class="font-semibold text-blue-600">${details.latest_attendance_summary}</span></p>
    //                 <p><strong class="font-medium text-gray-600">Trạng thái điểm danh:</strong> <span class="${isAttendanceOpen ? 'text-green-600 font-semibold' : 'text-gray-500'}">${details.attendance_status}</span></p>
    //             </div>
    //         </div>
    //         <div>
    //             <h4 class="text-lg font-semibold text-slate-700 mb-3">Lịch Dạy Trong Tuần</h4>
    //             <div class="overflow-x-auto border rounded-md"><table class="min-w-full text-sm">
    //                 <thead class="bg-gray-100"><tr><th class="px-3 py-2 text-left font-medium text-gray-600">Thứ</th><th class="px-3 py-2 text-left font-medium text-gray-600">Thời Gian</th><th class="px-3 py-2 text-left font-medium text-gray-600">Phòng</th></tr></thead>
    //                 <tbody>${details.schedules.map(s => {
    //                     console.log('Schedule:', s);
    //                     console.log('start time', s.start_time, 'end time:', s.end_time);
    //                      return `<tr class="border-b"><td class="px-3 py-2">${s.day}</td><td class="px-3 py-2">${extractTime(s.start_time)} - ${extractTime(s.end_time)}</td><td class="px-3 py-2">${s.room}</td></tr>`;
    //                 }).join('') || '<tr><td colspan="3" class="text-center p-3 text-gray-500">Không có lịch học.</td></tr>'}</tbody>
    //             </table></div>
    //         </div>
    //         <div>
    //             <h4 class="text-lg font-semibold text-slate-700 mb-3">Danh Sách Sinh Viên & Trạng Thái Buổi Gần Nhất (${studentList.length})</h4>
    //             <div class="overflow-x-auto max-h-60 border rounded-md"><table class="min-w-full text-sm">
    //                 <thead class="bg-gray-100 sticky top-0 z-10"><tr><th class="px-3 py-2 text-left font-medium text-gray-600">MSSV</th><th class="px-3 py-2 text-left font-medium text-gray-600">Họ Tên</th><th class="px-3 py-2 text-left font-medium text-gray-600">Trạng Thái</th><th class="px-3 py-2 text-left font-medium text-gray-600">Số Buổi Vắng</th></tr></thead>
    //                 <tbody>${studentList.map(s => {
    //                     const classDetails = details.students.find(student => student.student_code === s.student_code) || {};

    //                     return `
    //                     <tr class="border-b">
    //                         <td class="px-3 py-2">${s.student_code}</td>
    //                         <td class="px-3 py-2">${s.name}</td>
    //                         <td class="px-3 py-2 ${statusClassMap[s.status] || 'text-gray-500'}">${statusTextMap[s.status] || s.status}</td>
    //                         <td class="px-3 py-2">${classDetails.absent_count}</td>
    //                     </tr>`;
    //                 }).join('') || '<tr><td colspan="3" class="text-center p-3 text-gray-500">Không có sinh viên trong lớp.</td></tr>'}
    //                 </tbody>
    //             </table></div>
    //         </div>`;
        
    //     elements.classDetailsModalToggleAttendance.textContent = isAttendanceOpen ? 'Đóng Điểm Danh' : 'Bắt Đầu Điểm Danh';
    //     elements.classDetailsModalToggleAttendance.className = `px-4 py-2 text-sm font-medium text-white rounded-md smooth-transition ${isAttendanceOpen ? 'bg-yellow-500 hover:bg-yellow-600' : 'bg-green-500 hover:bg-green-600'}`;
    // }
    
    // --- EVENT HANDLERS ---
   
    function populateClassDetailsModal(details, studentList) {
        const daysName = ["Thứ hai", "Thứ ba", "Thứ tư", "Thứ năm", "Thứ sáu", "Thứ bảy", "Chủ nhật"];
        elements.classDetailsModalTitle.textContent = `Chi Tiết Lớp: ${details.subject_name} (${details.class_code})`;
        const isAttendanceOpen = details.attendance_status === "Đang mở";
        
        // --- BẢN ĐỒ TRẠNG THÁI CHO DROPDOWN ---
        const statusOptions = {
            PRESENT: 'Có mặt',
            ABSENT: 'Vắng',
            UNMARKED: 'Chưa điểm danh'
        };

        elements.classDetailsModalBody.innerHTML = `
            <div>
                <h4 class="text-lg font-semibold text-slate-700 mb-2">Thông Tin Chung</h4>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2 text-sm">
                    <p><strong class="font-medium text-gray-600">Mã lớp:</strong> ${details.class_code}</p>
                    <p><strong class="font-medium text-gray-600">Sĩ số:</strong> ${details.student_count} sinh viên</p>
                    <p><strong class="font-medium text-gray-600">Điểm danh (gần nhất):</strong> <span class="font-semibold text-blue-600">${details.latest_attendance_summary}</span></p>
                    <p><strong class="font-medium text-gray-600">Trạng thái điểm danh:</strong> <span class="${isAttendanceOpen ? 'text-green-600 font-semibold' : 'text-gray-500'}">${details.attendance_status}</span></p>
                </div>
            </div>
            <div>
                <h4 class="text-lg font-semibold text-slate-700 mb-3">Lịch Dạy Trong Tuần</h4>
                <div class="overflow-x-auto border rounded-md"><table class="min-w-full text-sm">
                    <thead class="bg-gray-100"><tr><th class="px-3 py-2 text-left font-medium text-gray-600">Thứ</th><th class="px-3 py-2 text-left font-medium text-gray-600">Thời Gian</th><th class="px-3 py-2 text-left font-medium text-gray-600">Phòng</th></tr></thead>
                    <tbody>${details.schedules.map(s => `
                        <tr class="border-b"><td class="px-3 py-2">${s.day}</td><td class="px-3 py-2">${extractTime(s.start_time)} - ${extractTime(s.end_time)}</td><td class="px-3 py-2">${s.room}</td></tr>`
                    ).join('') || '<tr><td colspan="3" class="text-center p-3 text-gray-500">Không có lịch học.</td></tr>'}</tbody>
                </table></div>
            </div>
            <div>
                <h4 class="text-lg font-semibold text-slate-700 mb-3">Danh Sách Sinh Viên & Trạng Thái Buổi Gần Nhất (${studentList.length})</h4>
                <div class="overflow-x-auto max-h-60 border rounded-md"><table class="min-w-full text-sm">
                    <thead class="bg-gray-100 sticky top-0 z-10"><tr><th class="px-3 py-2 text-left font-medium text-gray-600">MSSV</th><th class="px-3 py-2 text-left font-medium text-gray-600">Họ Tên</th><th class="px-3 py-2 text-left font-medium text-gray-600">Trạng Thái</th><th class="px-3 py-2 text-left font-medium text-gray-600">Số Buổi Vắng</th></tr></thead>
                    <tbody>
                    ${studentList.map(s => {
                        const studentDetails = details.students.find(student => student.student_code === s.student_code) || {};
                        // --- TẠO DROPDOWN THAY VÌ TEXT ---
                        return `
                        <tr class="border-b">
                            <td class="px-3 py-2">${s.student_code}</td>
                            <td class="px-3 py-2">${s.name}</td>
                            <td class="px-3 py-2">
                                <select data-student-id="${s.student_db_id}" class="manual-attendance-select form-select block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50 text-sm py-1">
                                    ${Object.entries(statusOptions).map(([value, text]) => `
                                        <option value="${value}" ${s.status === value ? 'selected' : ''}>${text}</option>
                                    `).join('')}
                                </select>
                            </td>
                            <td class="px-3 py-2 text-center">${studentDetails.absent_count || 0}</td>
                        </tr>`;
                    }).join('') || '<tr><td colspan="4" class="text-center p-3 text-gray-500">Không có sinh viên trong lớp.</td></tr>'}
                    </tbody>
                </table></div>
            </div>`;
        
        elements.classDetailsModalToggleAttendance.textContent = isAttendanceOpen ? 'Đóng Điểm Danh' : 'Bắt Đầu Điểm Danh';
        elements.classDetailsModalToggleAttendance.className = `px-4 py-2 text-sm font-medium text-white rounded-md smooth-transition ${isAttendanceOpen ? 'bg-yellow-500 hover:bg-yellow-600' : 'bg-green-500 hover:bg-green-600'}`;

        // --- GẮN SỰ KIỆN CHO CÁC DROPDOWN MỚI TẠO ---
        document.querySelectorAll('.manual-attendance-select').forEach(select => {
            select.addEventListener('change', handleManualUpdate);
        });
    }

    async function handleManualUpdate(event) {
        const selectElement = event.target;
        const studentId = selectElement.dataset.studentId;
        const newStatus = selectElement.value;
        const classId = currentViewingClassId;

        if (!classId || !studentId) {
            console.error("Không có classId hoặc studentId để cập nhật.");
            return;
        }

        // Vô hiệu hóa dropdown để tránh click nhiều lần
        selectElement.disabled = true;

        try {
            await api.updateManualAttendance(classId, studentId, newStatus);
            // Cập nhật lại toàn bộ modal để đảm bảo dữ liệu (như số buổi vắng) được đồng bộ
            // Đây là cách đơn giản và hiệu quả nhất
            await openClassDetailsModal(classId); 
        } catch (error) {
            alert(`Lỗi khi cập nhật điểm danh: ${error.message}`);
            // Nếu lỗi, tải lại modal để trả về trạng thái cũ
            await openClassDetailsModal(classId); 
        } finally {
            // Dù thành công hay thất bại, dropdown sẽ được kích hoạt lại khi modal được vẽ lại
        }
    }

    async function handleToggleAttendanceClick(classId, isFromQuickButton = false) {
        const button = isFromQuickButton ? document.getElementById('quickToggleAttendanceButton') : elements.classDetailsModalToggleAttendance;
        const originalContent = button.innerHTML;
        button.innerHTML = '<span>Đang xử lý...</span>';
        button.disabled = true;

        try {
            const details = await api.getClassDetails(classId);
            const action = details.attendance_status === "Đang mở" ? api.stopAttendance : api.startAttendance;
            const successMessage = details.attendance_status === "Đang mở" 
                ? `Đã đóng phiên điểm danh cho lớp ${details.subject_name}.`
                : `Đã bắt đầu phiên điểm danh cho lớp ${details.subject_name}.`;

            await action(classId);
            alert(successMessage);
            
            if (elements.classDetailsModal.classList.contains('active')) {
                openClassDetailsModal(classId); 
            }
            const managedClasses = await api.getManagedClasses();
            updateNextClassSection(managedClasses);
        } catch (error) {
            alert(`Lỗi: ${error.message}`);
            button.innerHTML = originalContent;
        } finally {
            button.disabled = false;
        }
    }
    
    function setupEventListeners() {
        
        const logoutHandler = () => {
            sessionStorage.removeItem('teacherAuthToken');
            sessionStorage.removeItem('teacherInfo');
            window.location.href = '/teacher/login'; 
        };
        document.getElementById('logoutButton').addEventListener('click', logoutHandler);

        elements.classDetailsModalToggleAttendance.addEventListener('click', () => {
            if(currentViewingClassId) handleToggleAttendanceClick(currentViewingClassId);
        });
        
        const closeModal = () => elements.classDetailsModal.classList.remove('active');
        elements.closeClassDetailsModalButton.addEventListener('click', closeModal);
        elements.classDetailsModalClose.addEventListener('click', closeModal);
        elements.classDetailsModal.addEventListener('click', (event) => {
            if (event.target === elements.classDetailsModal) closeModal();
        });

        document.querySelectorAll('a.nav-link, a.mobile-nav-link').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const targetId = this.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                     const navbarHeight = document.querySelector('nav').offsetHeight;
                     const elementPosition = targetElement.getBoundingClientRect().top;
                     const offsetPosition = elementPosition + window.pageYOffset - navbarHeight - 16; // 16px offset
                    window.scrollTo({ top: offsetPosition, behavior: 'smooth' });
                }
                if (this.classList.contains('mobile-nav-link')) {
                    elements.mobileMenu.classList.add('hidden');
                    elements.iconOpen.classList.remove('hidden');
                    elements.iconClose.classList.add('hidden');
                }
            });
        });
    }
    
    // --- MAIN EXECUTION ---
    async function main() {
        try {
            setupEventListeners();
            document.getElementById('currentYearFooter').textContent = new Date().getFullYear();
            
            const dashboardData = await api.getDashboardData();
            updateDashboardUI(dashboardData);

            const managedClasses = await api.getManagedClasses();
            updateManagedClassesTable(managedClasses);
            updateNextClassSection(managedClasses);

            const teacherId = dashboardData.teacher_info.id;
            if (teacherId) {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const socket = new WebSocket(`${protocol}//${window.location.host}/ws/${teacherId}`);

                socket.onopen = function(e) {
                    console.log("[open] Kết nối WebSocket đã được thiết lập.");
                };

                socket.onmessage = function(event) {
                    try {
                        const notification = JSON.parse(event.data);
                        if (notification.type === 'new_check_in') {
                            const message = `SV <b>${notification.student_name}</b> vừa điểm danh lớp <b>${notification.subject_name} (${notification.class_code})</b>.`;
                            showToast(message, 5000);
                        }
                    } catch (err) {
                        console.error("Lỗi xử lý tin nhắn WebSocket:", err);
                    }
                };

                socket.onclose = function(event) {
                    if (event.wasClean) {
                        console.log(`[close] Kết nối đã đóng sạch, code=${event.code} reason=${event.reason}`);
                    } else {
                        console.log('[close] Kết nối bị ngắt');
                    }
                };

                socket.onerror = function(error) {
                    console.error(`[error] ${error.message}`);
                };
            }

        } catch (error) {
            console.error('Lỗi khi tải dữ liệu trang:', error);
            document.body.innerHTML = `<div class="p-8 text-center text-red-600"><strong>Không thể tải dữ liệu từ server.</strong><br>Vui lòng thử lại sau. Lỗi: ${error.message}</div>`;
        }
    }

    main();
});