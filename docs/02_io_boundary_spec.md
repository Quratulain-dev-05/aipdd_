## 2. System Boundary, Actors, and Input/Output Specification

**Case study:** AI-powered video surveillance application that detects people and objects of interest and raises alerts.

### 2.1 System Boundary

| Inside the system boundary | Outside the system boundary |
|----------------------------|-----------------------------|
| Frame ingestion and decoding | Physical cameras and their firmware |
| Image preprocessing | Network infrastructure and the RTSP server |
| Model inference and post-processing | Model training (done offline) |
| Alert rule evaluation and event logging | SMS / email provider infrastructure |
| Operator dashboard and report generation | Building access-control hardware |

### 2.2 Primary Actors / Users

| Actor | Type | Role and Responsibilities | Interaction with System |
|-------|------|---------------------------|-------------------------|
| Security Operator | Human (primary user) | Monitors the live feed, acknowledges or dismisses alerts, queries past events. | Dashboard: views annotated stream, receives alerts, submits acknowledgements and queries. |
| System Administrator | Human | Configures streams, thresholds and alert rules; deploys model updates; manages users and audits logs. | Configuration interface and file/API access to config and model stores. |
| Camera / RTSP Source | External system | Supplies continuous video. | Pushes video over RTSP into ingestion. |
| Automated Trigger System | External system | Raises events (e.g. motion sensor, door contact) that wake the pipeline or raise detection sensitivity. | Sends trigger signals through an API or message queue. |
| Notification Service | External system | Delivers alerts by SMS, email or push. | Receives alert messages from the system. |

### 2.3 System Inputs

| ID | Input | Source | Format / Parameters | Rate / Size |
|----|-------|--------|---------------------|-------------|
| IN-01 | Live video stream | IP camera | RTSP over TCP/UDP, H.264 / H.265 | 15-30 FPS per camera |
| IN-02 | Frame resolution | Camera settings | 1280x720 (720p) default; up to 1920x1080 | About 2.7 MB per raw 720p frame |
| IN-03 | Sensor parameters | Camera / config | Exposure, day-night mode, field of view, camera ID | Static, loaded at start-up |
| IN-04 | Model weights | Administrator | `.pt` or `.onnx` file (e.g. YOLOv8n) | 6-50 MB |
| IN-05 | Detection thresholds | Configuration file | Confidence (default 0.25-0.5), NMS IoU (default 0.45) | Per deployment |
| IN-06 | Alert rules | Administrator | Watch-list labels, zones, schedules, severity | YAML / JSON |
| IN-07 | External trigger | Sensor / trigger system | Event message (type, camera ID, timestamp) | Event-driven |
| IN-08 | Operator commands | Dashboard | Acknowledge alert, query by date/camera/class | On demand |

### 2.4 System Outputs

| ID | Output | Destination | Format | Content |
|----|--------|-------------|--------|---------|
| OUT-01 | Bounding box coordinates | Downstream modules / API | List of `(x1, y1, x2, y2, class, confidence)` in original-frame pixels | Per frame |
| OUT-02 | Annotated live feed | Operator dashboard | MJPEG / WebRTC stream with drawn boxes and labels | At least 15 FPS |
| OUT-03 | Alert notifications | Operator, Notification Service | JSON message, SMS, email, dashboard pop-up | Rule name, severity, camera, time, snapshot |
| OUT-04 | Log entries | Event database and log files | SQLite / SQL rows, rotating text log | Timestamp, frame ID, rule, severity, detections |
| OUT-05 | Analytics reports | Operator, Administrator | CSV / PDF | Counts by class, hour and camera |
| OUT-06 | System health status | Administrator | JSON / dashboard widget | FPS, latency, CPU/RAM, stream connectivity |

### 2.5 Operational Constraints

| ID | Constraint | Limit |
|----|------------|-------|
| C-01 | Maximum memory footprint | 2 GB RAM per edge node (model plus buffers) |
| C-02 | Bandwidth per camera | 4 Mbps or less for 720p H.264; only metadata and alert snapshots go to the cloud (under 100 kbps average) |
| C-03 | End-to-end latency | 200 ms or less from capture to alert |
| C-04 | Throughput | At least 15 FPS per camera; up to 4 cameras per edge node |
| C-05 | Storage | Event log kept for 90 days; video clips kept 7 days; automatic rotation |
| C-06 | Power / thermal | 15 W or less on the edge device |
| C-07 | Compliance | Local privacy law; surveillance signage; access restricted by role |
| C-08 | Environment | Python 3.10+, runs offline once the model is deployed |
