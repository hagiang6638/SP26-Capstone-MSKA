"""
inference_video.py
==================
Inference một video (hoặc thư mục ảnh) → gloss sequence
dùng mô hình MSKA-SLR đã train trên Phoenix-2014T.
Keypoint extractor: MediaPipe (thay cho HRNet)

Usage:
    # Inference từ file video
    python inference_video.py \\
        --input path/to/video.mp4 \\
        --config configs/phoenix-2014t_s2g.yaml \\
        --checkpoint pretrained_models/Phoenix-2014T_SLR/best.pth

    # Inference từ thư mục ảnh (frames đã extract sẵn, định dạng Phoenix)
    python inference_video.py \\
        --input path/to/frame_folder/ \\
        --config configs/phoenix-2014t_s2g.yaml \\
        --checkpoint pretrained_models/Phoenix-2014T_SLR/best.pth

Cải tiến so với v1:
    [1] Dùng decord thay cv2 để đọc video + sample frames TRƯỚC khi extract keypoint
        → Tránh extract keypoint trên frame thừa, nhanh hơn đáng kể
    [2] Hỗ trợ cả folder ảnh (*.png/*.jpg) lẫn file video (.mp4, .avi, ...)
        → Tương thích với dữ liệu Phoenix (frames extract sẵn)
    [3] Tự động clean prefix '.module.' trong state_dict khi load checkpoint
        → Tránh lỗi khi checkpoint được lưu từ DataParallel (multi-GPU)

MediaPipe keypoint layout (553 points total, refine_landmarks=True):
    [0  – 32 ] Pose         (33 pts)
    [33 – 53 ] Left hand    (21 pts)  ← MediaPipe "Right" label (mirror)
    [54 – 74 ] Right hand   (21 pts)  ← MediaPipe "Left"  label (mirror)
    [75 – 552] Face         (478 pts) ← refine_landmarks=True → 478, không phải 468
"""

import argparse
import os
import cv2
import yaml
import torch
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import OrderedDict

# ── Detect MediaPipe version để chọn đúng API
#    < 0.10 : Legacy Solutions API  (mp.solutions.pose, ...)
#    >= 0.10 : Tasks API            (mediapipe.tasks.python.vision, ...)
_mp_ver = tuple(int(x) for x in mp.__version__.split('.')[:2])
MP_NEW_API = _mp_ver >= (0, 10)
print(f"[INFO] MediaPipe {mp.__version__} → {'Tasks API (>=0.10)' if MP_NEW_API else 'Legacy Solutions API (<0.10)'}")

if MP_NEW_API:
    from mediapipe.tasks import python as _mp_tasks_python
    from mediapipe.tasks.python import vision as _mp_tasks_vision
else:
    try:
        _MpPose     = mp.solutions.pose.Pose
        _MpHands    = mp.solutions.hands.Hands
        _MpFaceMesh = mp.solutions.face_mesh.FaceMesh
    except AttributeError as e:
        raise RuntimeError(
            f"MediaPipe {mp.__version__} không tìm thấy solutions API.\n"
            "Thử downgrade: pip install mediapipe==0.9.3"
        ) from e

# decord: đọc video nhanh hơn cv2, hỗ trợ random-access frame
try:
    from decord import VideoReader, cpu as decord_cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    print("[WARN] decord không được cài. Fallback sang cv2.VideoCapture.")
    print("       Cài bằng: pip install decord")

from model import SignLanguageModel
from Tokenizer import GlossTokenizer_S2G


# ============================================================
# CONSTANTS
# ============================================================
IMG_W, IMG_H = 720, 576   # kích thước resize frame (Phoenix-2014T)
CLIP_LEN      = 400       # max frames model có thể xử lý

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
IMAGE_EXTENSIONS  = {'.jpg', '.jpeg', '.png', '.bmp'}

# Thư mục chứa model files cho Tasks API (tự tạo, tự download)
MP_MODEL_DIR = Path(__file__).parent / '.mp_models'


# ============================================================
# Step 1 – Chọn frame index (uniform sampling – học từ SEN_CSLR)
# ============================================================
def get_sampled_indices(total_frames: int, clip_len: int = CLIP_LEN):
    """
    Uniform sampling: giữ đều các frame trải dài toàn video.
    Khác với center-crop (cũ), cách này bảo toàn thông tin đầu/cuối.
    Kết quả được căn chỉnh chia hết cho 4 (yêu cầu của DSTA-Net stride).
    """
    if total_frames <= clip_len:
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, clip_len, dtype=int)

    valid_len = len(indices)
    # Phải chia hết cho 4 (do 2× stride-2 trong DSTA-Net)
    if valid_len % 4 != 0:
        valid_len -= valid_len % 4
        indices = indices[:valid_len]

    return indices.tolist(), valid_len


# ============================================================
# Step 2a – Load frames từ VIDEO FILE (dùng decord nếu có)
# ============================================================
def load_frames_from_video(video_path: str, clip_len: int = CLIP_LEN):
    """
    Đọc video → sample indices TRƯỚC → chỉ decode đúng frame cần.
    Trả về list ảnh RGB shape (H, W, 3).
    """
    if HAS_DECORD:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        total = len(vr)
        indices, valid_len = get_sampled_indices(total, clip_len)
        print(f"[INFO] Video: {total} frames tổng → sample {valid_len} frames (decord)")
        batch = vr.get_batch(indices).asnumpy()   # (N, H, W, 3) – RGB
        frames = [batch[i] for i in range(len(batch))]
    else:
        # Fallback: cv2 đọc tuần tự rồi chọn
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video: {video_path}")
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        total = len(all_frames)
        indices, valid_len = get_sampled_indices(total, clip_len)
        frames = [all_frames[i] for i in indices]
        print(f"[INFO] Video: {total} frames tổng → sample {valid_len} frames (cv2)")

    return frames


# ============================================================
# Step 2b – Load frames từ FOLDER ẢNH (học từ SEN_CSLR)
# ============================================================
def load_frames_from_folder(folder_path: str, clip_len: int = CLIP_LEN):
    """
    Đọc thư mục chứa frames (*.png, *.jpg) đã extract sẵn.
    Hữu ích với dataset Phoenix (frames trích xuất thành ảnh riêng lẻ).
    """
    img_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])
    if not img_paths:
        raise RuntimeError(f"Không tìm thấy ảnh trong: {folder_path}")

    total = len(img_paths)
    indices, valid_len = get_sampled_indices(total, clip_len)
    print(f"[INFO] Folder: {total} ảnh tổng → sample {valid_len} frames")

    frames = []
    for i in indices:
        img = cv2.imdecode(
            np.fromfile(img_paths[i], dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


# ============================================================
# Step 2 – Dispatch: tự phát hiện video hay folder
# ============================================================
def load_frames(input_path: str, clip_len: int = CLIP_LEN):
    """
    Nhận vào đường dẫn → tự nhận dạng video file hay thư mục ảnh.
    """
    if os.path.isdir(input_path):
        return load_frames_from_folder(input_path, clip_len)
    ext = os.path.splitext(input_path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return load_frames_from_video(input_path, clip_len)
    raise ValueError(
        f"Input không hợp lệ: {input_path}\n"
        f"Hỗ trợ: thư mục ảnh hoặc video {VIDEO_EXTENSIONS}"
    )


# ============================================================
# Step 3 – MediaPipe estimators
# ============================================================
def _download_mp_models(model_dir: Path):
    """
    Auto-download model .task files cho Tasks API (MediaPipe >= 0.10).
    Chỉ download nếu file chưa tồn tại.
    """
    import urllib.request
    model_dir.mkdir(parents=True, exist_ok=True)
    models = {
        'pose_landmarker.task':
            'https://storage.googleapis.com/mediapipe-models/pose_landmarker'
            '/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
        'hand_landmarker.task':
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker'
            '/hand_landmarker/float16/latest/hand_landmarker.task',
        'face_landmarker.task':
            'https://storage.googleapis.com/mediapipe-models/face_landmarker'
            '/face_landmarker/float16/latest/face_landmarker.task',
    }
    for fname, url in models.items():
        dst = model_dir / fname
        if not dst.exists():
            print(f'[INFO] Downloading {fname} ...')
            urllib.request.urlretrieve(url, dst)
            print(f'[INFO] Saved → {dst}')
    return model_dir


def build_mediapipe_estimators():
    """
    Khởi tạo estimators phù hợp với version MediaPipe đang cài:
      >= 0.10 : Tasks API  – PoseLandmarker / HandLandmarker / FaceLandmarker
      <  0.10 : Legacy API – Pose / Hands / FaceMesh (mp.solutions.*)
    """
    if MP_NEW_API:
        model_dir = _download_mp_models(MP_MODEL_DIR)
        RunningMode = _mp_tasks_vision.RunningMode

        pose_est = _mp_tasks_vision.PoseLandmarker.create_from_options(
            _mp_tasks_vision.PoseLandmarkerOptions(
                base_options=_mp_tasks_python.BaseOptions(
                    model_asset_path=str(model_dir / 'pose_landmarker.task')),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
        )
        hand_est = _mp_tasks_vision.HandLandmarker.create_from_options(
            _mp_tasks_vision.HandLandmarkerOptions(
                base_options=_mp_tasks_python.BaseOptions(
                    model_asset_path=str(model_dir / 'hand_landmarker.task')),
                running_mode=RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
            )
        )
        face_est = _mp_tasks_vision.FaceLandmarker.create_from_options(
            _mp_tasks_vision.FaceLandmarkerOptions(
                base_options=_mp_tasks_python.BaseOptions(
                    model_asset_path=str(model_dir / 'face_landmarker.task')),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
        )
    else:  # Legacy API
        pose_est = _MpPose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )
        hand_est = _MpHands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        face_est = _MpFaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # → 478 điểm
            min_detection_confidence=0.5
        )
    return pose_est, hand_est, face_est


def extract_keypoints_frame(img_rgb, pose_est, hand_est, face_est):
    """
    Trả về list 553 điểm: [x_pixel, y_pixel, score]
    Layout: [pose(33) | left_hand(21) | right_hand(21) | face(478)]

    Convention handedness (QUAN TRỌNG - khác nhau giữa 2 API):
      Legacy API  : label "Right" = tay phải trong ảnh = tay TRÁI người ký
                    → đặt vào slot left_hand (33-53)
      Tasks API   : label "Left"  = tay trái người ký (nhìn từ góc của người ký)
                    → đặt vào slot left_hand (33-53)
    """
    H, W = img_rgb.shape[:2]
    keypoints = []

    if MP_NEW_API:
        # ── Tasks API (MediaPipe >= 0.10) ──
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # POSE (33)
        pose_res = pose_est.detect(mp_image)
        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks[0]:
                vis = lm.visibility if lm.visibility is not None else 1.0
                keypoints.append([lm.x * W, lm.y * H, vis])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 33)

        # HANDS
        hand_res = hand_est.detect(mp_image)
        left_lms, right_lms = None, None
        if hand_res.hand_landmarks and hand_res.handedness:
            for lms, hd in zip(hand_res.hand_landmarks, hand_res.handedness):
                # Tasks API: "Left" = subject's left hand → slot left_hand (33-53)
                label = hd[0].category_name
                if label == 'Left':
                    left_lms = lms
                else:  # 'Right' = subject's right hand → slot right_hand (54-74)
                    right_lms = lms

        if left_lms:
            for lm in left_lms:
                keypoints.append([lm.x * W, lm.y * H, 1.0])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 21)

        if right_lms:
            for lm in right_lms:
                keypoints.append([lm.x * W, lm.y * H, 1.0])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 21)

        # FACE (478 — FaceLandmarker Tasks API luôn trả về 478 điểm)
        face_res = face_est.detect(mp_image)
        if face_res.face_landmarks:
            for lm in face_res.face_landmarks[0]:
                keypoints.append([lm.x * W, lm.y * H, 1.0])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 478)

    else:
        # ── Legacy Solutions API (MediaPipe < 0.10) ──

        # POSE (33)
        pose_res = pose_est.process(img_rgb)
        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks.landmark:
                keypoints.append([lm.x * W, lm.y * H, lm.visibility])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 33)

        # HANDS
        hand_res = hand_est.process(img_rgb)
        left_hand, right_hand = None, None
        if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
            for lm, handedness in zip(
                hand_res.multi_hand_landmarks,
                hand_res.multi_handedness
            ):
                # Legacy API: "Right" = camera's right = người ký tay TRÁI
                label = handedness.classification[0].label
                if label == 'Right':
                    left_hand = lm   # slot 33-53
                else:
                    right_hand = lm  # slot 54-74

        if left_hand:
            for lm in left_hand.landmark:
                keypoints.append([lm.x * W, lm.y * H, 1.0])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 21)

        if right_hand:
            for lm in right_hand.landmark:
                keypoints.append([lm.x * W, lm.y * H, 1.0])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 21)

        # FACE (478 với refine_landmarks=True)
        face_res = face_est.process(img_rgb)
        if face_res.multi_face_landmarks:
            for lm in face_res.multi_face_landmarks[0].landmark:
                keypoints.append([lm.x * W, lm.y * H, 1.0])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 478)

    assert len(keypoints) == 553, f"Expected 553 keypoints, got {len(keypoints)}"
    return keypoints


def extract_keypoints_from_frames(frames):
    """
    Nhận list ảnh RGB (đã sample) → chạy MediaPipe từng frame
    → tensor (T, 553, 3)
    """
    pose_est, hand_est, face_est = build_mediapipe_estimators()
    all_kp = []
    for i, frame in enumerate(frames):
        frame_resized = cv2.resize(frame, (IMG_W, IMG_H))
        kp = extract_keypoints_frame(frame_resized, pose_est, hand_est, face_est)
        all_kp.append(kp)
        if (i + 1) % 50 == 0:
            print(f"  [MediaPipe] {i+1}/{len(frames)} frames xử lý...")

    pose_est.close(); hand_est.close(); face_est.close()
    print(f"[INFO] Đã extract keypoints: {len(all_kp)} frames × 553 points")
    return torch.tensor(all_kp, dtype=torch.float32)   # (T, 553, 3)


# ============================================================
# Step 4 – Normalize keypoints (giống datasets.py collate_fn)
# ============================================================
def normalize_keypoints(keypoints):
    """
    keypoints: tensor (C=3, T, V=553)
    Áp dụng cùng normalize như trong datasets.py augment_preprocess_inputs()
    """
    kp = keypoints.clone()
    kp[0] /= IMG_W                  # x / W  → [0, 1]
    kp[1] = IMG_H - kp[1]          # flip y
    kp[1] /= IMG_H                  # y / H  → [0, 1]
    kp[:2] = (kp[:2] - 0.5) / 0.5  # → [-1, 1]
    return kp


# ============================================================
# Step 5 – Build src_input dict cho model
# ============================================================
def build_src_input(keypoint_tensor, device):
    """
    keypoint_tensor: (T, V=553, C=3)  (output của extract_keypoints_from_frames)
    → Trả về dict src_input sẵn sàng đưa vào model.recognition_network()
    """
    T = keypoint_tensor.shape[0]

    # permute → (C=3, T, V=553)  (giống datasets.py __getitem__)
    kp = keypoint_tensor.permute(2, 0, 1).float()

    # Normalize
    kp = normalize_keypoints(kp)

    # Thêm batch dim → (1, 3, T, 553)
    kp_batch = kp.unsqueeze(0)

    # Tính sequence length sau 2× stride-2 của DSTA-Net
    src_length = torch.tensor([T])
    new_src_lengths = (((src_length - 1) // 2) + 1)
    new_src_lengths = (((new_src_lengths - 1) // 2) + 1)

    max_len = int(new_src_lengths.max().item())
    mask = torch.zeros(1, 1, max_len, dtype=torch.bool)
    mask[0, :, :new_src_lengths[0]] = True

    src_input = {
        'name':            ['inference_video'],
        'keypoint':        kp_batch.to(device),
        'mask':            mask.to(device),
        'new_src_lengths': new_src_lengths.to(device),
        'gloss':           [''],
        'gloss_input': {
            'gloss_labels': torch.zeros(1, 1, dtype=torch.long).to(device),
            'gls_lengths':  torch.ones(1, dtype=torch.long).to(device),
        },
        'src_length': src_length.to(device),
    }
    return src_input


# ============================================================
# Step 6 – Load model (với clean .module. – học từ SEN_CSLR)
# ============================================================
def load_model(config, checkpoint_path, device):
    """
    Load SignLanguageModel và state_dict từ checkpoint.
    Tự động xóa prefix '.module.' nếu checkpoint lưu từ DataParallel (multi-GPU).
    """
    model = SignLanguageModel(
        cfg=config,
        args=argparse.Namespace(device=str(device), run=None)
    )

    raw = torch.load(checkpoint_path, map_location='cpu')

    # Checkpoint có thể dùng key 'model' hoặc 'model_state_dict'
    if 'model' in raw:
        state_dict = raw['model']
    elif 'model_state_dict' in raw:
        state_dict = raw['model_state_dict']
    else:
        raise KeyError(f"Checkpoint không chứa key 'model' hay 'model_state_dict'.\n"
                       f"Keys có sẵn: {list(raw.keys())}")

    # Xóa prefix '.module.' (DataParallel artifact) – học từ SEN_CSLR
    state_dict = OrderedDict(
        (k.replace('.module', ''), v) for k, v in state_dict.items()
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.to(device)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    return model


# ============================================================
# Step 7 – Inference + CTC decode
# ============================================================
@torch.no_grad()
def run_inference(model, src_input, tokenizer, beam_size=5):
    output = model.recognition_network(src_input)

    # Dùng ensemble logits (tổng hợp 4 stream: left, right, body, fuse)
    gloss_logits = output['ensemble_last_gloss_logits']

    ctc_decode = model.recognition_network.decode(
        gloss_logits=gloss_logits,
        beam_size=beam_size,
        input_lengths=src_input['new_src_lengths']
    )

    batch_gloss = tokenizer.convert_ids_to_tokens(ctc_decode)
    return batch_gloss[0]   # sample duy nhất trong batch


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='MSKA Inference: Video/Folder → Gloss sequence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python inference_video.py --input video.mp4 --config configs/phoenix-2014t_s2g.yaml --checkpoint best.pth
  python inference_video.py --input frames_folder/ --config configs/phoenix-2014t_s2g.yaml --checkpoint best.pth
        """
    )
    parser.add_argument('--input',      required=True,
                        help='Đường dẫn video (.mp4/.avi/...) hoặc thư mục ảnh frames')
    parser.add_argument('--config',     required=True,
                        help='File config YAML (vd: configs/phoenix-2014t_s2g.yaml)')
    parser.add_argument('--checkpoint', required=True,
                        help='File checkpoint .pth')
    parser.add_argument('--beam_size',  type=int, default=5,
                        help='Beam size cho CTC decode (default: 5)')
    parser.add_argument('--device',     default='cuda',
                        help='Thiết bị: cuda hoặc cpu (default: cuda)')
    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # Config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Tokenizer
    tokenizer = GlossTokenizer_S2G(config['gloss'])
    print(f"[INFO] Vocab size: {len(tokenizer)}")

    # Model
    print("[INFO] Đang load model...")
    model = load_model(config, args.checkpoint, device)

    # Load + sample frames
    print(f"[INFO] Đang load input: {args.input}")
    frames = load_frames(args.input, clip_len=CLIP_LEN)

    # Extract keypoints (chỉ trên frames đã sample)
    print(f"[INFO] Đang extract MediaPipe keypoints trên {len(frames)} frames...")
    keypoint_tensor = extract_keypoints_from_frames(frames)

    # Build model input
    src_input = build_src_input(keypoint_tensor, device)

    # Inference
    print("[INFO] Đang chạy inference...")
    gloss_seq = run_inference(model, src_input, tokenizer, beam_size=args.beam_size)

    # Output
    print("\n" + "=" * 60)
    print(f"INPUT : {args.input}")
    print(f"GLOSS : {' '.join(gloss_seq)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
