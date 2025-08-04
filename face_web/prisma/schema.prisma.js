// Prisma schema for AI-Face application

generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

// Models

model Student {
  id                String             @id @default(cuid())
  studentCode       String             @unique
  name              String
  password          String
  createdAt         DateTime           @default(now())
  updatedAt         DateTime           @updatedAt
  enrollments       Enrollment[]
  attendanceRecords AttendanceRecord[]
  @@map("students")
}

model Teacher {
  id          String        @id @default(cuid())
  teacherCode String        @unique
  name        String
  password    String
  createdAt   DateTime      @default(now())
  updatedAt   DateTime      @updatedAt
  classes     CourseClass[]
  @@map("teachers")
}

model Course {
  id          String        @id @default(cuid())
  courseCode  String        @unique
  courseName  String
  description String?
  classes     CourseClass[]
  @@map("courses")
}

model CourseClass {
  id                 String              @id @default(cuid())
  classCode          String              @unique
  semester           String
  isActive           Boolean             @default(true)
  courseId           String
  teacherId          String
  course             Course              @relation(fields: [courseId], references: [id])
  teacher            Teacher             @relation(fields: [teacherId], references: [id])
  enrollments        Enrollment[]
  schedules          ClassSchedule[]
  attendanceSessions AttendanceSession[]
  @@map("classes")
}

model Enrollment {
  id             String      @id @default(cuid())
  enrollmentDate DateTime    @default(now())
  studentId      String
  classId        String
  student        Student     @relation(fields: [studentId], references: [id])
  course_class   CourseClass @relation(fields: [classId], references: [id])
  @@unique([studentId, classId])
  @@map("enrollments")
}

model ClassSchedule {
  id           String      @id @default(cuid())
  dayOfWeek    Int
  startTime    String
  endTime      String
  room         String?
  classId      String
  course_class CourseClass @relation(fields: [classId], references: [id])
  @@map("class_schedules")
}

model AttendanceSession {
  id           String             @id @default(cuid())
  sessionDate  DateTime
  isOpen       Boolean            @default(false)
  openedAt     DateTime?
  closedAt     DateTime?
  classId      String
  course_class CourseClass        @relation(fields: [classId], references: [id])
  records      AttendanceRecord[]
  @@unique([classId, sessionDate])
  @@map("attendance_sessions")
}

model AttendanceRecord {
  id          String           @id @default(cuid())
  status      AttendanceStatus
  checkInTime DateTime?
  notes       String?
  sessionId   String
  studentId   String
  session     AttendanceSession @relation(fields: [sessionId], references: [id])
  student     Student           @relation(fields: [studentId], references: [id])
  @@unique([sessionId, studentId])
  @@map("attendance_records")
}

enum AttendanceStatus {
  UNMARKED
  PRESENT
  ABSENT
}
