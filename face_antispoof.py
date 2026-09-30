"""
Single-file inference: SCRFD (det_500g, uint8 quantized) face detector
+ MiniFASNetV2 (int8 quantized) anti-spoofing, on a webcam stream.

- Uses tflite_runtime only (no tensorflow fallback).
- Loads the VX NPU delegate (/usr/lib/libvx_delegate.so) by default and
  falls back to CPU if it can't be loaded. Use --cpu to skip the NPU.

Usage:
    python face_antispoof.py
    python face_antispoof.py --cpu --camera 1
"""

import argparse
import time

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


# ============================================================
# CONFIG
# ============================================================
DET_MODEL = "det_500g_640x640_full_quantized.tflite"
ANTI_SPOOF_MODEL = "models/MiniFASNetV2_full_integer_quant_v2.tflite"

NPU_DELEGATE_PATH = "/usr/lib/libvx_delegate.so"

# SCRFD
INPUT_SIZE = 640
FEAT_STRIDES = [8, 16, 32]
NUM_ANCHORS = 2

# MiniFASNet
ANTI_INPUT_SIZE = 80
ANTI_CROP_SCALE = 2.7
REAL_CLASS_ID = 1  # 0 = fake, 1 = real, 2 = fake


# ============================================================
# INTERPRETER LOADING (tflite_runtime + VX delegate)
# ============================================================
def load_interpreter(model_path, use_npu=True, num_threads=4):
    delegates = []

    if use_npu:
        try:
            delegates.append(tflite.load_delegate(NPU_DELEGATE_PATH))
            print(f"[NPU] Delegate loaded: {NPU_DELEGATE_PATH}")
        except (ValueError, OSError) as e:
            print(f"[NPU] Failed to load delegate ({e}) -> falling back to CPU")

    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=delegates,
        num_threads=None if delegates else num_threads,
    )
    interpreter.allocate_tensors()
    return interpreter


def dequantize(arr, quant):
    scale, zero_point = quant
    if scale == 0:
        return arr.astype(np.float32)
    return (arr.astype(np.float32) - zero_point) * scale


# ============================================================
# SCRFD DETECTOR
# ============================================================
class FaceDetector:
    def __init__(self, model_path, use_npu=True):
        self.interpreter = load_interpreter(model_path, use_npu)
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_details = self.interpreter.get_output_details()
        self._anchor_cache = {}

    def _anchors(self, stride):
        if stride not in self._anchor_cache:
            h = w = INPUT_SIZE // stride
            centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            if NUM_ANCHORS > 1:
                centers = np.repeat(centers, NUM_ANCHORS, axis=0)
            self._anchor_cache[stride] = centers
        return self._anchor_cache[stride]

    @staticmethod
    def _distance2bbox(points, d):
        return np.stack(
            [points[:, 0] - d[:, 0], points[:, 1] - d[:, 1],
             points[:, 0] + d[:, 2], points[:, 1] + d[:, 3]],
            axis=-1,
        )

    @staticmethod
    def _distance2kps(points, d):
        out = []
        for i in range(0, d.shape[1], 2):
            out.append(points[:, 0] + d[:, i])
            out.append(points[:, 1] + d[:, i + 1])
        return np.stack(out, axis=-1)

    @staticmethod
    def _nms(dets, thresh):
        x1, y1, x2, y2, scores = dets.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= thresh)[0] + 1]
        return keep

    @staticmethod
    def _letterbox(img_rgb):
        h, w = img_rgb.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img_rgb, (new_w, new_h))
        padded = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        return padded, scale

    def detect(self, frame_bgr, score_thresh=0.5, nms_thresh=0.4):
        """Returns (bboxes Nx5 [x1,y1,x2,y2,score], kpss Nx5x2) or (None, None)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        padded, scale = self._letterbox(rgb)

        # uint8 pixels go in directly (normalization is baked into quant params)
        self.interpreter.set_tensor(self.input_index, padded[None].astype(np.uint8))
        self.interpreter.invoke()

        # Group outputs by (rows, cols): rows -> stride, cols -> scores(1)/bbox(4)/kps(10)
        by_rows = {}
        for od in self.output_details:
            raw = self.interpreter.get_tensor(od["index"])
            arr = dequantize(raw, od["quantization"]).reshape(-1, od["shape"][-1])
            by_rows.setdefault(arr.shape[0], {})[arr.shape[1]] = arr

        all_scores, all_boxes, all_kps = [], [], []

        for stride in FEAT_STRIDES:
            rows = (INPUT_SIZE // stride) ** 2 * NUM_ANCHORS
            if rows not in by_rows:
                continue

            group = by_rows[rows]
            scores = group[1].reshape(-1)
            pos = np.where(scores >= score_thresh)[0]
            if pos.size == 0:
                continue

            centers = self._anchors(stride)
            boxes = self._distance2bbox(centers, group[4] * stride)
            kps = self._distance2kps(centers, group[10] * stride)
            kps = kps.reshape(kps.shape[0], -1, 2)

            all_scores.append(scores[pos])
            all_boxes.append(boxes[pos])
            all_kps.append(kps[pos])

        if not all_scores:
            return None, None

        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes) / scale
        kps = np.concatenate(all_kps) / scale

        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = self._nms(dets, nms_thresh)
        return dets[keep], kps[keep]


# ============================================================
# MINIFASNET ANTI-SPOOF
# ============================================================
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def crop_face(image, bbox_xywh, scale=ANTI_CROP_SCALE, out_size=ANTI_INPUT_SIZE):
    """MiniFASNet-style enlarged crop around the face box."""
    src_h, src_w = image.shape[:2]
    x, y, box_w, box_h = bbox_xywh

    if box_w <= 0 or box_h <= 0:
        return None, None

    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x + box_w / 2, y + box_h / 2

    x1 = max(0, int(cx - new_w / 2))
    y1 = max(0, int(cy - new_h / 2))
    x2 = min(src_w - 1, int(cx + new_w / 2))
    y2 = min(src_h - 1, int(cy + new_h / 2))

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = image[y1:y2 + 1, x1:x2 + 1]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop, (x1, y1, x2, y2)


class AntiSpoof:
    def __init__(self, model_path, use_npu=True):
        self.interpreter = load_interpreter(model_path, use_npu)
        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]

        self.in_index = inp["index"]
        self.in_dtype = inp["dtype"]
        self.in_scale, self.in_zp = inp["quantization"]
        self.out_index = out["index"]
        self.out_quant = out["quantization"]

        print("\n========== MiniFASNetV2 ==========")
        print("Input :", inp["name"], inp["shape"], inp["dtype"].__name__,
              "scale=", self.in_scale, "zp=", self.in_zp)
        print("Output:", out["name"], out["shape"], out["dtype"].__name__,
              "scale=", self.out_quant[0], "zp=", self.out_quant[1])
        print("=" * 30)

    def _quantize_input(self, face_bgr):
        # Original MiniFASNet preprocessing: BGR, float32, 0..255 (no /255, no RGB)
        x = face_bgr.astype(np.float32)[None]  # NHWC

        if self.in_dtype == np.float32 or self.in_scale == 0:
            return x.astype(self.in_dtype)

        info = np.iinfo(self.in_dtype)
        q = np.round(x / self.in_scale + self.in_zp)
        return np.clip(q, info.min, info.max).astype(self.in_dtype)

    def predict(self, frame_bgr, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        face, crop_box = crop_face(frame_bgr, (x1, y1, x2 - x1, y2 - y1))
        if face is None:
            return None

        self.interpreter.set_tensor(self.in_index, self._quantize_input(face))
        self.interpreter.invoke()

        raw = self.interpreter.get_tensor(self.out_index)
        logits = dequantize(raw, self.out_quant).reshape(1, -1)
        probs = softmax(logits)[0]

        class_id = int(np.argmax(probs))
        return {
            "label": "Real" if class_id == REAL_CLASS_ID else "Fake",
            "class_id": class_id,
            "confidence": float(probs[class_id]),
            "probabilities": probs,
            "crop_box": crop_box,
        }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="skip NPU delegate")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--det-model", default=DET_MODEL)
    parser.add_argument("--spoof-model", default=ANTI_SPOOF_MODEL)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.4)
    args = parser.parse_args()

    use_npu = not args.cpu

    print("=" * 60)
    print("Loading SCRFD face detector")
    print("=" * 60)
    detector = FaceDetector(args.det_model, use_npu)

    print("\n" + "=" * 60)
    print("Loading MiniFASNetV2 anti-spoofing")
    print("=" * 60)
    spoof = AntiSpoof(args.spoof_model, use_npu)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nPress 'q' to quit.\n")

    prev = time.time()
    fps = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        now = time.time()
        dt = now - prev
        prev = now
        if dt > 0:
            inst = 1.0 / dt
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

        bboxes, _kpss = detector.detect(frame, args.score, args.nms)

        if bboxes is not None:
            for det in bboxes:
                x1, y1, x2, y2 = det[:4].astype(int)
                det_score = float(det[4])

                result = spoof.predict(frame, (x1, y1, x2, y2))
                if result is None:
                    continue

                label = result["label"]
                probs = result["probabilities"]
                color = (0, 255, 0) if label == "Real" else (0, 0, 255)

                # detector box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # anti-spoof input region (yellow)
                cx1, cy1, cx2, cy2 = result["crop_box"]
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 1)

                cv2.putText(frame, f"{label}: {result['confidence']:.2f}",
                            (x1, max(25, y1 - 10)), font, 0.7, color, 2, cv2.LINE_AA)

                prob_text = " ".join(f"C{i}:{p:.2f}" for i, p in enumerate(probs))
                cv2.putText(frame, prob_text,
                            (x1, min(frame.shape[0] - 10, y2 + 22)),
                            font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(frame, f"Face: {det_score:.2f}",
                            (x1, min(frame.shape[0] - 30, y2 + 42)),
                            font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), font, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "SCRFD 500G + MiniFASNetV2 INT8", (20, 65), font, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("SCRFD 500G + MiniFASNetV2 Anti-Spoofing", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam stopped.")


if __name__ == "__main__":
    main()
