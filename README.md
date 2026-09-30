# Anti-Spoofing Face Detector (MiniFASNetV2 + SCRFD 500G)

Real-time face anti-spoofing pipeline that combines a lightweight **SCRFD 500G** face detector with **MiniFASNetV2** liveness classification, running fully quantized (INT8/UINT8) TFLite models on a live webcam feed. Designed for edge deployment with optional **VeriSilicon VX NPU delegate** acceleration and automatic CPU fallback.

## Overview

The pipeline works in two stages per frame:

1. **Face Detection (SCRFD 500G)** — a quantized, anchor-based detector (`det_500g_640x640_full_quantized.tflite`) locates faces and 5-point landmarks in the frame using multi-stride feature maps (strides 8/16/32) with NMS-based box refinement.
2. **Liveness Classification (MiniFASNetV2)** — for each detected face, an enlarged crop is fed into a quantized MiniFASNetV2 model (`models/MiniFASNetV2_full_integer_quant_v2.tflite`) to classify the face as **Real** or **Fake**, with a per-class confidence score.

Results are overlaid on the live video feed: detection box, anti-spoofing crop region, predicted label, confidence, and class probabilities.

## Features

- Single-file inference script — no training code, no heavy framework dependencies.
- Uses `tflite_runtime` only (no full TensorFlow required).
- Automatic **NPU delegate** loading (`/usr/lib/libvx_delegate.so`) with graceful fallback to CPU.
- Fully vectorized NumPy pre/post-processing (letterboxing, anchor decoding, NMS, dequantization).
- Configurable detection/NMS thresholds and camera index via CLI flags.
- Real-time FPS overlay.

## Project Structure

```
.
├── face_antispoof.py                          # Main inference script
├── det_500g_640x640_full_quantized.tflite      # SCRFD 500G face detector (quantized)
└── models/
    └── MiniFASNetV2_full_integer_quant_v2.tflite  # MiniFASNetV2 anti-spoof classifier (quantized)
```

## Requirements

- Python 3.7+
- [`tflite_runtime`](https://www.tensorflow.org/lite/guide/python)
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:

```bash
pip install numpy opencv-python
pip install tflite-runtime
```

> **Note:** For NPU acceleration, the VeriSilicon VX delegate library (`libvx_delegate.so`) must be present on the target device (e.g. NXP i.MX 8M Plus / similar NPU-enabled boards). If unavailable, the script automatically falls back to CPU inference.

## Usage

Run with default settings (webcam index 0, NPU delegate if available):

```bash
python face_antispoof.py
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--cpu` | off | Skip NPU delegate and force CPU inference |
| `--camera` | `0` | Webcam device index |
| `--det-model` | `det_500g_640x640_full_quantized.tflite` | Path to the face detector model |
| `--spoof-model` | `models/MiniFASNetV2_full_integer_quant_v2.tflite` | Path to the anti-spoofing model |
| `--score` | `0.5` | Face detection confidence threshold |
| `--nms` | `0.4` | NMS IoU threshold for face detection |

Example:

```bash
python face_antispoof.py --cpu --camera 1 --score 0.6
```

Press **`q`** to quit the video window.

## Output

Each detected face is annotated with:

- **Green box** — detected as **Real**
- **Red box** — detected as **Fake**
- **Yellow box** — the exact crop region fed into MiniFASNetV2
- Label + confidence, per-class probabilities, and face detection score
- Live FPS counter

## Models

| Model | Task | Precision | Input Size |
|---|---|---|---|
| SCRFD 500G | Face detection + landmarks | UINT8 | 640×640 |
| MiniFASNetV2 | Binary/ternary liveness classification | INT8 | 80×80 |

`REAL_CLASS_ID` (default `1`) in `face_antispoof.py` maps the model's output classes to `Real`/`Fake` labels — adjust if using a differently trained checkpoint.

## License

No license specified. Add a `LICENSE` file if you intend to distribute or open-source this project.
