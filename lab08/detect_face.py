import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    if box_scores is None or len(box_scores) == 0:
        return np.empty((0, box_scores.shape[1] if hasattr(box_scores, 'shape') else 5), dtype=np.float32)

    if box_scores.shape[1] < 5:
        raise ValueError("box_scores must have shape (N, 5) with [x1, y1, x2, y2, score]")

    scores = box_scores[:, -1]
    order = np.argsort(scores)
    if candidate_size > 0 and order.size > candidate_size:
        order = order[-candidate_size:]

    picked = []
    while order.size > 0:
        current = order[-1]
        picked.append(current)

        if top_k > 0 and len(picked) >= top_k:
            break
        if order.size == 1:
            break

        order = order[:-1]
        if order.size == 0:
            break

        active = box_scores[current, :4]
        remaining_boxes = box_scores[order, :4]

        inter_x1 = np.maximum(active[0], remaining_boxes[:, 0])
        inter_y1 = np.maximum(active[1], remaining_boxes[:, 1])
        inter_x2 = np.minimum(active[2], remaining_boxes[:, 2])
        inter_y2 = np.minimum(active[3], remaining_boxes[:, 3])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        area_active = max(0.0, active[2] - active[0]) * max(0.0, active[3] - active[1])
        area_remaining = np.maximum(0.0, remaining_boxes[:, 2] - remaining_boxes[:, 0]) * np.maximum(0.0, remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = area_active + area_remaining - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = order[iou <= iou_threshold]

    return box_scores[np.array(picked, dtype=np.int64)]


def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size

    resized = image.resize((320, 240), Image.BILINEAR)
    resized_array = np.asarray(resized, dtype=np.float32)
    resized_array = (resized_array - 127.0) / 128.0
    resized_array = np.transpose(resized_array, (2, 0, 1))[None, :, :, :]

    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=available_providers or ["CPUExecutionProvider"])
    scores, boxes = session.run(None, {"input": resized_array})

    face_scores = scores[0, :, 1]
    valid = face_scores >= conf_threshold
    boxes = boxes[0, valid]
    face_scores = face_scores[valid]

    if boxes.size == 0:
        image.save(output_path)
        return np.empty((0, 5), dtype=np.float32)

    boxes = boxes.astype(np.float32)
    x1 = boxes[:, 0] * original_width
    y1 = boxes[:, 1] * original_height
    x2 = boxes[:, 2] * original_width
    y2 = boxes[:, 3] * original_height

    box_scores = np.column_stack([x1, y1, x2, y2, face_scores])
    kept = hard_nms(box_scores, iou_threshold=0.3, top_k=-1, candidate_size=200)

    image_draw = ImageDraw.Draw(image)
    for x1_i, y1_i, x2_i, y2_i, score in kept:
        x1_i = max(0, min(x1_i, original_width))
        y1_i = max(0, min(y1_i, original_height))
        x2_i = max(0, min(x2_i, original_width))
        y2_i = max(0, min(y2_i, original_height))
        image_draw.rectangle((x1_i, y1_i, x2_i, y2_i), outline="green", width=2)
        image_draw.text((x1_i, max(0, y1_i - 18)), f"{score:.2f}", fill="green")

    image.save(output_path)
    return kept

# Edge detection
def non_maximum_suppression(magnitude, angle):
    magnitude = np.asarray(magnitude, dtype=np.float32)
    angle = np.asarray(angle, dtype=np.float32)
    output = np.zeros_like(magnitude, dtype=np.float32)

    angle_deg = np.mod(np.degrees(angle), 180.0)
    height, width = magnitude.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            theta = angle_deg[i, j]
            value = magnitude[i, j]

            if 0 <= theta < 22.5 or 157.5 <= theta <= 180:
                neighbors = (magnitude[i, j - 1], magnitude[i, j + 1])
            elif 22.5 <= theta < 67.5:
                neighbors = (magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])
            elif 67.5 <= theta < 112.5:
                neighbors = (magnitude[i - 1, j], magnitude[i + 1, j])
            else:
                neighbors = (magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])

            if value >= neighbors[0] and value >= neighbors[1]:
                output[i, j] = value

    return output


def hysteresis_thresholding(img, low_thresh, high_thresh):
    img = np.asarray(img, dtype=np.float32)
    output = np.zeros_like(img, dtype=np.uint8)

    strong = img >= high_thresh
    weak = (img >= low_thresh) & (img < high_thresh)
    output[strong] = 255

    coords = np.argwhere(weak)
    for i, j in coords:
        neighborhood = img[max(0, i - 1):min(img.shape[0], i + 2), max(0, j - 1):min(img.shape[1], j + 2)]
        if np.any(neighborhood >= high_thresh):
            output[i, j] = 255

    return output


def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    image = Image.open(image_path).convert("L")
    grayscale = np.asarray(image, dtype=np.float32)
    blurred = gaussian_filter(grayscale, sigma=1.4)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=np.float32)

    gx = convolve(blurred, sobel_x)
    gy = convolve(blurred, sobel_y)
    magnitude = np.hypot(gx, gy)
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255.0
    angle = np.arctan2(gy, gx)

    thinned = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thinned, low, high)

    edge_image = Image.fromarray(edges, mode="L")
    edge_image.save(output_path)
    return edges

# CHANGE THE NAME OF THE .jpg TO TEST FOR A DIFFERENT IMAGE.
if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")
