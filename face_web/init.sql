--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13
-- Dumped by pg_dump version 15.13

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: AttendanceStatus; Type: TYPE; Schema: public; Owner: user_attendance
--

CREATE TYPE public."AttendanceStatus" AS ENUM (
    'UNMARKED',
    'PRESENT',
    'ABSENT'
);


ALTER TYPE public."AttendanceStatus" OWNER TO user_attendance;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: _prisma_migrations; Type: TABLE; Schema: public; Owner: user_attendance
--

CREATE TABLE public._prisma_migrations (
    id character varying(36) NOT NULL,
    checksum character varying(64) NOT NULL,
    finished_at timestamp with time zone,
    migration_name character varying(255) NOT NULL,
    logs text,
    rolled_back_at timestamp with time zone,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    applied_steps_count integer DEFAULT 0 NOT NULL
);


ALTER TABLE public._prisma_migrations OWNER TO user_attendance;

--
-- Name: attendance_records; Type: TABLE; Schema: public; Owner: user_attendance
--

CREATE TABLE public.attendance_records (
    id text NOT NULL,
    status public."AttendanceStatus" NOT NULL,
    "checkInTime" timestamp(3) without time zone,
    notes text,
    "sessionId" text NOT NULL,
    "studentId" text NOT NULL
);


ALTER TABLE public.attendance_records OWNER TO user_attendance;

--
-- Name: attendance_sessions; Type: TABLE; Schema: public; Owner: user_attendance
--

CREATE TABLE public.attendance_sessions (
    id text NOT NULL,
    "sessionDate" timestamp(3) without time zone NOT NULL,
    "isOpen" boolean DEFAULT false NOT NULL,
    "openedAt" timestamp(3) without time zone,
    "closedAt" timestamp(3) without time zone,
    "classId" text NOT NULL
);


ALTER TABLE public.attendance_sessions OWNER TO user_attendance;

--
-- Name: class_schedules; Type: TABLE; Schema: public; Owner: user_attendance
--

CREATE TABLE public.class_schedules (
    id text NOT NULL,
    "dayOfWeek" integer NOT NULL,
    "startTime" text NOT NULL,
    "endTime" text NOT NULL,
    room text,
    "classId" text NOT NULL
);


ALTER TABLE public.class_schedules OWNER TO user_attendance;

--
-- Name: classes; Type: TABLE; Schema: public; Owner: user_attendance
--

CREATE TABLE public.classes (
    id text NOT NULL,
    "classCode" text NOT NULL,
    semester text NOT NULL,
    "isActive" boolean DEFAULT true NOT NULL,
    "courseId" text NOT NULL,
    "teacherId" text NOT NULL
);


ALTER TABLE public.classes OWNER TO user_attendance;

--
-- Name: courses; Type: TABLE; Schema: public; Owner: user_attendance
--

CREATE TABLE public.courses (
    id text NOT NULL,
    "courseCode" text NOT NULL,
    "courseName" text NOT NULL,
    description text
);


ALTER TABLE public.courses OWNER TO user_attendance;

--
-- Name: enrollments; Type: TABLE; Schema: public; Owner: user_attendance
--

CREATE TABLE public.enrollments (
    id text NOT NULL,
    "enrollmentDate" timestamp(3) without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    "studentId" text NOT NULL,
    "classId" text NOT NULL
);


ALTER TABLE public.enrollments OWNER TO user_attendance;

--
-- Name: students; Type: TABLE; Schema: public; Owner: user_attendance
--

CREATE TABLE public.students (
    id text NOT NULL,
    "studentCode" text NOT NULL,
    name text NOT NULL,
    password text NOT NULL,
    "createdAt" timestamp(3) without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    "updatedAt" timestamp(3) without time zone NOT NULL
);


ALTER TABLE public.students OWNER TO user_attendance;

--
-- Name: teachers; Type: TABLE; Schema: public; Owner: user_attendance
--

CREATE TABLE public.teachers (
    id text NOT NULL,
    "teacherCode" text NOT NULL,
    name text NOT NULL,
    password text NOT NULL,
    "createdAt" timestamp(3) without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    "updatedAt" timestamp(3) without time zone NOT NULL
);


ALTER TABLE public.teachers OWNER TO user_attendance;

--
-- Data for Name: _prisma_migrations; Type: TABLE DATA; Schema: public; Owner: user_attendance
--

COPY public._prisma_migrations (id, checksum, finished_at, migration_name, logs, rolled_back_at, started_at, applied_steps_count) FROM stdin;
bda74452-9e20-46ed-aaf2-2b5900b36e9d	216544e5995b4cd7a331eea9de5cfe953a1994358731e7f64704b9e98022bb75	2025-07-27 08:01:22.251935+00	20250727080122_init	\N	\N	2025-07-27 08:01:22.113987+00	1
ffd035c9-e370-4550-b057-58c959e93077	dc71e3ad46a40c0de0b3c2d2202f5b0f0c72e132b58733a872531f124b55a2fa	2025-07-27 09:47:54.689047+00	20250727094754_init	\N	\N	2025-07-27 09:47:54.66602+00	1
\.


--
-- Data for Name: attendance_records; Type: TABLE DATA; Schema: public; Owner: user_attendance
--

COPY public.attendance_records (id, status, "checkInTime", notes, "sessionId", "studentId") FROM stdin;
cmdmy777a0002rajwu8btb5a0	ABSENT	\N	\N	cmdmy776y0001rajwtwtotprs	cmdle5zwm0000h55ismlsll0w
cmdo3xuro0002mcvi03my898u	PRESENT	2025-07-29 08:55:33.647	\N	cmdo3xur80001mcvia2urls2c	cmdle5zwm0000h55ismlsll0w
cmdoj2ihk0002h2d9ts486gfw	PRESENT	2025-07-29 13:02:58.232	Cập nhật thủ công bởi giáo viên	cmdoj2ih50001h2d9z9s29hhy	cmdle5zwm0000h55ismlsll0w
cmdsoh7b60002auumb2qy14wr	PRESENT	2025-08-01 10:32:11.799	\N	cmdsoh7as0001auumah2fe4fe	cmdle5zwm0000h55ismlsll0w
\.


--
-- Data for Name: attendance_sessions; Type: TABLE DATA; Schema: public; Owner: user_attendance
--

COPY public.attendance_sessions (id, "sessionDate", "isOpen", "openedAt", "closedAt", "classId") FROM stdin;
cmdmy776y0001rajwtwtotprs	2025-07-28 10:11:19.209	f	2025-07-28 10:11:19.209	2025-07-28 10:14:05.1	cmdlepnjv000ah55if7jytra7
cmdo3xur80001mcvia2urls2c	2025-07-29 05:39:47.05	f	2025-07-29 05:39:47.05	2025-07-29 08:56:36.826	cmdlepnjv000ah55if7jytra7
cmdoj2ih50001h2d9z9s29hhy	2025-07-29 12:43:18.656	f	2025-07-29 12:43:18.656	\N	cmdlepnjv000ah55if7jytra7
cmdsoh7as0001auumah2fe4fe	2025-08-01 10:25:46.793	f	2025-08-01 10:25:46.793	2025-08-01 10:35:39.703	cmdlepnjv000ah55if7jytra7
\.


--
-- Data for Name: class_schedules; Type: TABLE DATA; Schema: public; Owner: user_attendance
--

COPY public.class_schedules (id, "dayOfWeek", "startTime", "endTime", room, "classId") FROM stdin;
cmdlu84lq0003h5irsxb2ijuf	3	1970-01-01 07:00:00	1970-01-01 10:35:00	G504	cmdlu6cbh0000h5irn2wongkz
cmdlu84lr0004h5ir2vrgm64m	3	1970-01-01 12:00:00	1970-01-01 15:35:00	G605	cmdlu6cbj0001h5irvyg29itq
cmdlu84lr0005h5irws6u9d69	3	1970-01-01 16:00:00	1970-01-01 20:00:00	G502	cmdlu6cbk0002h5iru0g3d8xh
cmdlf5p8b000dh55in0t02bsh	4	1970-01-01 07:00:00	1970-01-01 10:35:00	G503	cmdlepnjv000ah55if7jytra7
cmdlf5p8c000eh55iss2g78a1	4	1970-01-01 12:00:00	1970-01-01 15:35:00	G601	cmdlepnjw000bh55il8v3aiuy
cmdlf5p8c000fh55izzhzfzkz	4	1970-01-01 16:00:00	1970-01-01 20:00:00	G605	cmdlepnjx000ch55ipyp5l76p
\.


--
-- Data for Name: classes; Type: TABLE DATA; Schema: public; Owner: user_attendance
--

COPY public.classes (id, "classCode", semester, "isActive", "courseId", "teacherId") FROM stdin;
cmdlepnjv000ah55if7jytra7	01010008640001	HK1_2025_2026	t	cmdlekvkf0002h55ipylcmea5	cmdle9wvu0001h55irvu58bik
cmdlu6cbh0000h5irn2wongkz	01010008690001	HK1_2025_2026	t	cmdlekvkg0003h55i00u72fqa	cmdle9wvu0001h55irvu58bik
cmdlu6cbj0001h5irvyg29itq	01010008690002	HK1_2025_2026	t	cmdlekvkg0003h55i00u72fqa	cmdle9wvu0001h55irvu58bik
cmdlu6cbk0002h5iru0g3d8xh	01010008690003	HK1_2025_2026	t	cmdlekvkg0003h55i00u72fqa	cmdle9wvu0001h55irvu58bik
cmdlepnjw000bh55il8v3aiuy	01010008640002	HK1_2025_2026	t	cmdlekvkf0002h55ipylcmea5	cmdle9wvu0001h55irvu58bik
cmdlepnjx000ch55ipyp5l76p	01010008640003	HK1_2025_2026	t	cmdlekvkf0002h55ipylcmea5	cmdle9wvu0001h55irvu58bik
\.


--
-- Data for Name: courses; Type: TABLE DATA; Schema: public; Owner: user_attendance
--

COPY public.courses (id, "courseCode", "courseName", description) FROM stdin;
cmdlekvkf0002h55ipylcmea5	0101000864	Đồ án chuyên ngành CNTT	Thực hiện một dự án phần mềm hoặc nghiên cứu hoàn chỉnh để tốt nghiệp, áp dụng các kiến thức đã học.
cmdlekvkg0003h55i00u72fqa	0101000869	Xử lý ảnh và thị giác máy tính	Nghiên cứu các thuật toán và kỹ thuật để máy tính có thể "nhìn" và hiểu được nội dung từ hình ảnh, video.
cmdlekvkg0004h55iozfzvwbp	0101000872	Lập trình Python	Giới thiệu về ngôn ngữ lập trình Python, cú pháp, và cách ứng dụng để giải quyết các bài toán thực tế.
cmdlekvkg0005h55inakahpff	0101000857	Hệ quản trị cơ sở dữ liệu	Tìm hiểu về cách thiết kế, quản lý và truy vấn các hệ thống cơ sở dữ liệu quan hệ như SQL Server, PostgreSQL.
cmdlekvkg0006h55i4d4zsn5l	0101000862	Phân tích thiết kế hệ thống	Cung cấp các phương pháp luận và công cụ để phân tích yêu cầu và thiết kế một hệ thống thông tin hiệu quả.
cmdlekvkg0007h55i64dswnnv	0101000867	Máy học	Khám phá các mô hình và thuật toán cho phép máy tính tự học hỏi từ dữ liệu mà không cần lập trình tường minh.
cmdlekvkg0008h55ietpohgot	0101000868	Lập trình mạng	Nghiên cứu các giao thức mạng và cách xây dựng các ứng dụng có khả năng giao tiếp qua mạng máy tính.
cmdlekvkh0009h55it09ub7cb	0101000967	Xây dựng ứng dụng thương mại điện tử	Hướng dẫn quy trình xây dựng một trang web hoặc ứng dụng thương mại điện tử hoàn chỉnh, từ giao diện đến xử lý thanh toán.
\.


--
-- Data for Name: enrollments; Type: TABLE DATA; Schema: public; Owner: user_attendance
--

COPY public.enrollments (id, "enrollmentDate", "studentId", "classId") FROM stdin;
cmdlf699q000gh55i7naxymk9	2025-07-27 08:30:56.366	cmdle5zwm0000h55ismlsll0w	cmdlepnjv000ah55if7jytra7
cmdlu8j520006h5irlc2soorx	2025-07-27 15:32:36.71	cmdle5zwm0000h55ismlsll0w	cmdlu6cbh0000h5irn2wongkz
cmdod34xl0000h5iyulsnr3ud	2025-07-29 09:55:50.074	cmdle5zwm0000h55ismlsll0w	cmdlepnjw000bh55il8v3aiuy
\.


--
-- Data for Name: students; Type: TABLE DATA; Schema: public; Owner: user_attendance
--

COPY public.students (id, "studentCode", name, password, "createdAt", "updatedAt") FROM stdin;
cmdle5zwm0000h55ismlsll0w	2331540042	Vũ Thái Bình Dương	123	2025-07-27 08:02:44.614	2025-07-27 08:02:20.772
cmdokvi6b0001h5iy27s0a5v8	2331540003	Giang Vạn Lộc	123	2025-07-29 13:33:50.916	2025-07-29 13:39:10.689
\.


--
-- Data for Name: teachers; Type: TABLE DATA; Schema: public; Owner: user_attendance
--

COPY public.teachers (id, "teacherCode", name, password, "createdAt", "updatedAt") FROM stdin;
cmdle9wvu0001h55irvu58bik	2254810032	Nguyễn Thái Sơn	123	2025-07-27 08:05:47.323	2025-07-27 08:06:15.753
\.


--
-- Name: _prisma_migrations _prisma_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public._prisma_migrations
    ADD CONSTRAINT _prisma_migrations_pkey PRIMARY KEY (id);


--
-- Name: attendance_records attendance_records_pkey; Type: CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.attendance_records
    ADD CONSTRAINT attendance_records_pkey PRIMARY KEY (id);


--
-- Name: attendance_sessions attendance_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.attendance_sessions
    ADD CONSTRAINT attendance_sessions_pkey PRIMARY KEY (id);


--
-- Name: class_schedules class_schedules_pkey; Type: CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.class_schedules
    ADD CONSTRAINT class_schedules_pkey PRIMARY KEY (id);


--
-- Name: classes classes_pkey; Type: CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.classes
    ADD CONSTRAINT classes_pkey PRIMARY KEY (id);


--
-- Name: courses courses_pkey; Type: CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.courses
    ADD CONSTRAINT courses_pkey PRIMARY KEY (id);


--
-- Name: enrollments enrollments_pkey; Type: CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.enrollments
    ADD CONSTRAINT enrollments_pkey PRIMARY KEY (id);


--
-- Name: students students_pkey; Type: CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT students_pkey PRIMARY KEY (id);


--
-- Name: teachers teachers_pkey; Type: CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.teachers
    ADD CONSTRAINT teachers_pkey PRIMARY KEY (id);


--
-- Name: attendance_records_sessionId_studentId_key; Type: INDEX; Schema: public; Owner: user_attendance
--

CREATE UNIQUE INDEX "attendance_records_sessionId_studentId_key" ON public.attendance_records USING btree ("sessionId", "studentId");


--
-- Name: attendance_sessions_classId_sessionDate_key; Type: INDEX; Schema: public; Owner: user_attendance
--

CREATE UNIQUE INDEX "attendance_sessions_classId_sessionDate_key" ON public.attendance_sessions USING btree ("classId", "sessionDate");


--
-- Name: classes_classCode_key; Type: INDEX; Schema: public; Owner: user_attendance
--

CREATE UNIQUE INDEX "classes_classCode_key" ON public.classes USING btree ("classCode");


--
-- Name: courses_courseCode_key; Type: INDEX; Schema: public; Owner: user_attendance
--

CREATE UNIQUE INDEX "courses_courseCode_key" ON public.courses USING btree ("courseCode");


--
-- Name: enrollments_studentId_classId_key; Type: INDEX; Schema: public; Owner: user_attendance
--

CREATE UNIQUE INDEX "enrollments_studentId_classId_key" ON public.enrollments USING btree ("studentId", "classId");


--
-- Name: students_studentCode_key; Type: INDEX; Schema: public; Owner: user_attendance
--

CREATE UNIQUE INDEX "students_studentCode_key" ON public.students USING btree ("studentCode");


--
-- Name: teachers_teacherCode_key; Type: INDEX; Schema: public; Owner: user_attendance
--

CREATE UNIQUE INDEX "teachers_teacherCode_key" ON public.teachers USING btree ("teacherCode");


--
-- Name: attendance_records attendance_records_sessionId_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.attendance_records
    ADD CONSTRAINT "attendance_records_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES public.attendance_sessions(id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: attendance_records attendance_records_studentId_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.attendance_records
    ADD CONSTRAINT "attendance_records_studentId_fkey" FOREIGN KEY ("studentId") REFERENCES public.students(id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: attendance_sessions attendance_sessions_classId_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.attendance_sessions
    ADD CONSTRAINT "attendance_sessions_classId_fkey" FOREIGN KEY ("classId") REFERENCES public.classes(id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: class_schedules class_schedules_classId_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.class_schedules
    ADD CONSTRAINT "class_schedules_classId_fkey" FOREIGN KEY ("classId") REFERENCES public.classes(id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: classes classes_courseId_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.classes
    ADD CONSTRAINT "classes_courseId_fkey" FOREIGN KEY ("courseId") REFERENCES public.courses(id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: classes classes_teacherId_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.classes
    ADD CONSTRAINT "classes_teacherId_fkey" FOREIGN KEY ("teacherId") REFERENCES public.teachers(id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: enrollments enrollments_classId_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.enrollments
    ADD CONSTRAINT "enrollments_classId_fkey" FOREIGN KEY ("classId") REFERENCES public.classes(id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- Name: enrollments enrollments_studentId_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user_attendance
--

ALTER TABLE ONLY public.enrollments
    ADD CONSTRAINT "enrollments_studentId_fkey" FOREIGN KEY ("studentId") REFERENCES public.students(id) ON UPDATE CASCADE ON DELETE RESTRICT;


--
-- PostgreSQL database dump complete
--

