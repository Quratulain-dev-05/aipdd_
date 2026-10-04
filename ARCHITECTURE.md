# System Design Specification: AI Object Detection and Analytics System

**Course:** AI Project Design and Development (AI-316), Air University Islamabad
**Lab:** 02 - System Requirements & Software Architecture for AI Projects
**Instructor:** Farhan Zafar Kayani

## Table of Contents
1. [Functional & Non-Functional Requirements](#1-functional--non-functional-requirements)
2. [System Boundary, Actors, and Input/Output Specification](#2-system-boundary-actors-and-inputoutput-specification)
3. [Data-Flow Diagrams](#3-data-flow-diagrams)
4. [Modular Software Architecture](#4-modular-software-architecture)
5. [Traceability and Repository Layout](#5-traceability-and-repository-layout)

---

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

---

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

---

## 3. Data-Flow Diagrams

Notation: **rectangle** = external entity, **circle** = process, **cylinder** = data store, **arrow** = labelled data flow. The diagrams are written in Mermaid (rendered by GitHub) and exported as PNG files in `docs/diagrams/`.

### 3.1 Level 0 - Context Diagram

```mermaid
flowchart LR
    CAM["Camera / RTSP Source"]
    OP["Security Operator"]
    ADM["Administrator"]
    NOTIF["Notification Service (SMS / Email / Dashboard)"]

    SYS(("0<br/>AI Object Detection<br/>and Analytics System"))

    CAM -- "Raw video stream (RTSP, H.264)" --> SYS
    ADM -- "Configuration, model updates, alert rules" --> SYS
    OP -- "Alert acknowledgements, report queries" --> SYS
    SYS -- "Live annotated feed, analytics reports" --> OP
    SYS -- "Alert messages" --> NOTIF
    SYS -- "System health status, audit logs" --> ADM
```

![DFD Level 0](docs/diagrams/dfd_level0.png)

### 3.2 Level 1 - Detailed Diagram

```mermaid
flowchart TB
    CAM["Camera / RTSP Source"]
    OP["Security Operator"]
    ADM["Administrator"]
    NOTIF["Notification Service"]

    P1(("1.0<br/>Ingest Video"))
    P2(("2.0<br/>Preprocess Frames"))
    P3(("3.0<br/>Run Model Inference"))
    P4(("4.0<br/>Post-process Detections"))
    P5(("5.0<br/>Raise Alerts and Log"))
    P6(("6.0<br/>Report and Display"))

    D1[("D1 Configuration Store")]
    D2[("D2 Model Weights Store")]
    D3[("D3 Detection and Event Log DB")]
    D4[("D4 Alert Rules Store")]

    CAM -- "Raw video stream" --> P1
    P1 -- "Decoded frames + timestamp" --> P2
    P2 -- "Normalised tensors (640x640)" --> P3
    P3 -- "Raw predictions (boxes, scores, classes)" --> P4
    P4 -- "Filtered detections (NMS, threshold)" --> P5
    P4 -- "Annotated frames" --> P6

    ADM -- "Settings" --> D1
    ADM -- "Trained weights (.pt / .onnx)" --> D2
    ADM -- "Alert rules" --> D4
    D1 -- "Stream URL, resolution, thresholds" --> P1
    D1 -- "Preprocessing parameters" --> P2
    D2 -- "Loaded model" --> P3
    D4 -- "Trigger conditions" --> P5

    P5 -- "Event records" --> D3
    P5 -- "Alert messages" --> NOTIF
    D3 -- "Historical events" --> P6
    P6 -- "Live feed, analytics reports" --> OP
    OP -- "Acknowledgements, queries" --> P6
    P6 -- "Health status, audit logs" --> ADM
```

![DFD Level 1](docs/diagrams/dfd_level1.png)

### 3.3 Process Descriptions

| Process | Name | Input | Output | Implemented by |
|---------|------|-------|--------|----------------|
| 1.0 | Ingest Video | Raw RTSP stream, D1 settings | Decoded frames with timestamp | `DataIngestion` |
| 2.0 | Preprocess Frames | Frames, D1 parameters | Normalised 640x640 tensors | `ImagePreprocessor` |
| 3.0 | Run Model Inference | Tensors, D2 weights | Raw predictions (boxes, scores, classes) | `ModelInferenceEngine` |
| 4.0 | Post-process Detections | Raw predictions | Filtered detections (confidence threshold, NMS), annotated frames | `ModelInferenceEngine` (NMS, thresholds) |
| 5.0 | Raise Alerts and Log | Filtered detections, D4 rules | Event records, alert messages | `AlertLogger` |
| 6.0 | Report and Display | Annotated frames, D3 history, operator queries | Live feed, reports, health status | Dashboard (future work) |

### 3.4 Data Stores

| Store | Name | Contents | Technology |
|-------|------|----------|------------|
| D1 | Configuration Store | Stream URL, resolution, thresholds, preprocessing parameters | `configs/*.yaml` |
| D2 | Model Weights Store | Trained detector weights | `models/*.pt`, `models/*.onnx` |
| D3 | Detection and Event Log DB | Timestamped events and detections | SQLite (`logs/events.db`), `logs/app.log` |
| D4 | Alert Rules Store | Watch-list labels, zones, severity | `configs/alert_rules.yaml` |

---

## 4. Modular Software Architecture

The system is split into four independent modules plus a pipeline orchestrator. Each module depends only on the shared data types in `src/data_types.py`, so any module can be replaced (for example a different camera source or detector) without touching the others.

```mermaid
flowchart LR
    A["DataIngestion"] -- "Frame" --> B["ImagePreprocessor"]
    B -- "PreprocessedFrame" --> C["ModelInferenceEngine"]
    C -- "List of Detection" --> D["AlertLogger"]
    D -- "List of Alert" --> E[("Event DB / Notifications")]
```

### 4.1 Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| `DataIngestion` | `src/data_ingestion.py` | Open a webcam, video file or RTSP stream; return timestamped frames; reconnect on failure; release resources. |
| `ImagePreprocessor` | `src/image_preprocessor.py` | Letterbox-resize to the model input size, convert BGR to RGB, scale to [0, 1], reorder to NCHW, keep scale/padding for mapping boxes back. |
| `ModelInferenceEngine` | `src/model_inference_engine.py` | Load weights, warm up, run detection, apply confidence threshold and NMS, return detections in original-frame pixels. |
| `AlertLogger` | `src/alert_logger.py` | Evaluate alert rules on detections, persist events to SQLite and a log file, call the notifier. |
| `DetectionPipeline` | `src/pipeline.py` | Connect the modules, run the frame loop, guarantee cleanup. |
| shared types | `src/data_types.py` | `Frame`, `PreprocessedFrame`, `Detection`, `Alert` dataclasses. |

### 4.2 Shared Data Types

| Class | Fields |
|-------|--------|
| `Frame` | `image: np.ndarray` (H x W x 3, BGR uint8), `frame_id: int`, `timestamp: datetime`, `source_id: str` |
| `PreprocessedFrame` | `tensor: np.ndarray` (1 x 3 x H x W, float32), `original_shape`, `scale: float`, `pad`, `frame: Frame` |
| `Detection` | `class_id: int`, `label: str`, `confidence: float`, `bbox: (x1, y1, x2, y2)` |
| `Alert` | `rule_name`, `severity`, `message`, `frame_id`, `timestamp`, `detections: List[Detection]` |

### 4.3 Class and Method Specification

#### DataIngestion (implements `BaseDataIngestion`)

| Method | Signature | Input | Returns |
|--------|-----------|-------|---------|
| constructor | `__init__(source: int or str, source_id: str = "cam0", reconnect_attempts: int = 3)` | Camera index, file path or RTSP URL | - |
| `open` | `open() -> bool` | - | True if the source opened |
| `read_frame` | `read_frame() -> Optional[Frame]` | - | Next `Frame`, or `None` at end of stream |
| `frames` | `frames() -> Iterator[Frame]` | - | Generator over all frames |
| `is_opened` | `is_opened() -> bool` | - | Connection state |
| `close` | `close() -> None` | - | - |

#### ImagePreprocessor

| Method | Signature | Input | Returns |
|--------|-----------|-------|---------|
| constructor | `__init__(target_size: Tuple[int, int] = (640, 640), pad_value: int = 114)` | Model input (H, W) | - |
| `process` | `process(frame: Frame) -> PreprocessedFrame` | One frame | Tensor plus scale/pad metadata |
| `process_batch` | `process_batch(frames: list) -> list` | List of frames | List of `PreprocessedFrame` |

#### ModelInferenceEngine (implements `BaseInferenceEngine`)

| Method | Signature | Input | Returns |
|--------|-----------|-------|---------|
| constructor | `__init__(weights_path, conf_threshold=0.25, iou_threshold=0.45, device="cpu")` | Model path and thresholds | - |
| `load_model` | `load_model() -> None` | - | - |
| `warmup` | `warmup(runs: int = 3) -> None` | Number of dummy runs | - |
| `predict` | `predict(item: PreprocessedFrame) -> List[Detection]` | One preprocessed frame | Detections in original-frame pixels |

#### AlertLogger

| Method | Signature | Input | Returns |
|--------|-----------|-------|---------|
| constructor | `__init__(db_path, log_file, watch_labels=None, min_confidence=0.5, notifier=None)` | Storage paths, rule parameters | - |
| `evaluate` | `evaluate(frame: Frame, detections: List[Detection]) -> List[Alert]` | Frame and detections | Triggered alerts (may be empty) |
| `log_event` | `log_event(alert: Alert) -> int` | One alert | Database row id |
| `send_notification` | `send_notification(alert: Alert) -> bool` | One alert | True if a notifier was called |
| `handle` | `handle(frame, detections) -> List[Alert]` | Frame and detections | Alerts (evaluated, logged, notified) |
| `close` | `close() -> None` | - | - |

#### DetectionPipeline

| Method | Signature | Returns |
|--------|-----------|---------|
| constructor | `__init__(ingestion, preprocessor, engine, alert_logger)` | - |
| `run` | `run(max_frames: Optional[int] = None) -> int` | Number of frames processed |

### 4.4 Usage Example

```python
from src.data_ingestion import DataIngestion
from src.image_preprocessor import ImagePreprocessor
from src.model_inference_engine import ModelInferenceEngine
from src.alert_logger import AlertLogger
from src.pipeline import DetectionPipeline

pipeline = DetectionPipeline(
    DataIngestion("rtsp://user:pass@192.168.1.10/stream1"),
    ImagePreprocessor((640, 640)),
    ModelInferenceEngine("models/yolov8n.pt", conf_threshold=0.4),
    AlertLogger(watch_labels=["person"]),
)
pipeline.run(max_frames=300)
```

---

## 5. Traceability and Repository Layout

### 5.1 Requirement Traceability

| Requirement | Addressed by |
|-------------|--------------|
| FR-01 Face/object detection | `ModelInferenceEngine.predict` (process 3.0 / 4.0) |
| FR-03 Attendance / event logging | `AlertLogger.log_event` (process 5.0, store D3) |
| FR-04 Synchronisation | `AlertLogger` local SQLite store plus a future sync job |
| NFR-01 Frame rate | `ModelInferenceEngine.warmup`, lightweight YOLOv8n weights |
| NFR-02 Accuracy | Confidence and IoU thresholds in `ModelInferenceEngine` |
| NFR-04 Privacy | Only metadata stored in D3; raw frames not persisted |
| C-01 Memory footprint | Small model, frame-by-frame processing (no frame queue growth) |
| NFR-05 Maintainability | Abstract base classes and config-driven model path |

### 5.2 Repository Layout

```
aipdd_project/
|-- ARCHITECTURE.md
|-- docs/
|   |-- 01_requirements.md
|   |-- 02_io_boundary_spec.md
|   `-- diagrams/        (dfd_level0.mmd/.png, dfd_level1.mmd/.png)
|-- src/
|   |-- __init__.py
|   |-- data_types.py
|   |-- data_ingestion.py
|   |-- image_preprocessor.py
|   |-- model_inference_engine.py
|   |-- alert_logger.py
|   `-- pipeline.py
|-- configs/  data/  models/  notebooks/  logs/
`-- requirements.txt
```

### 5.3 Known Limitations

- Process 6.0 (dashboard and reports) is specified but not implemented in this lab.
- Alert rules are limited to label and confidence; zone and schedule rules are future work.
- The database synchronisation job (FR-04) is specified but not implemented.
