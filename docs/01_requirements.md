## 1. Functional & Non-Functional Requirements

**Case study:** Smart Automated Attendance System. A camera at a classroom door captures students, the system detects and recognises faces, marks attendance automatically and syncs records to a central database.

### 1.1 Functional Requirements

| ID | Requirement | Acceptance Criterion | Priority |
|----|-------------|----------------------|----------|
| FR-01 | **Face detection:** The system shall detect all faces visible in each processed video frame. | Detection latency of 100 ms or less per frame on the target edge device; recall of 95% or more on the validation set. | High |
| FR-02 | **Face recognition:** The system shall match each detected face against the enrolled student database and return a student ID or "unknown". | Top-1 identification accuracy of 97% or more on enrolled students; unknown faces are never assigned an ID. | High |
| FR-03 | **Attendance logging:** The system shall log one attendance record (student ID, timestamp, class session, camera ID, confidence) the first time a student is recognised in a session. | No duplicate record per student per session; every record carries a timestamp accurate to 1 second. | High |
| FR-04 | **Database synchronisation:** The system shall store records locally and synchronise them with the central server, including after network outages. | Records appear on the server within 30 seconds of connectivity; zero records lost after a 1-hour offline period. | High |
| FR-05 | **Enrolment and reporting:** An administrator shall be able to enrol or remove students and export attendance reports per class and date range. | Enrolment of a new student completes in under 2 minutes; reports export as CSV and PDF. | Medium |

### 1.2 Non-Functional Requirements

| ID | Category | Requirement | Measurable Threshold | Priority |
|----|----------|-------------|----------------------|----------|
| NFR-01 | Performance | The pipeline shall sustain real-time processing of the live camera stream. | At least 15 FPS end-to-end on the edge device; no more than 200 ms from frame capture to logged result. | High |
| NFR-02 | Accuracy | The recognition model shall meet minimum quality thresholds before deployment. | Face detection mAP@0.5 of at least 0.90; recognition false-accept rate of at most 1%; false-reject rate of at most 3%. | High |
| NFR-03 | Resource / Power | The system shall run on a low-power edge device (e.g. Jetson Nano or Raspberry Pi 5 class). | Peak power of 15 W or less; RAM usage of 2 GB or less; continuous operation for 8 hours without thermal throttling. | Medium |
| NFR-04 | Data Privacy & Security | Biometric data shall be protected and used only for attendance. | Face embeddings encrypted at rest (AES-256); traffic over TLS 1.2 or higher; raw video not stored beyond 24 hours; role-based access; student consent recorded. | High |
| NFR-05 | Reliability & Maintainability | The system shall recover automatically from failures and be maintainable. | 99% uptime during class hours; automatic restart within 30 seconds of a crash; model replaceable without code changes (config only). | Medium |
