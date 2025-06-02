import cv2
import numpy as np
import tensorflow as tf

# Load model dari folder saved_model (pastikan pathnya bener ya)
detect_fn = tf.saved_model.load("ssd_mobilenet_v2/saved_model")

# Label hewan yang mau dideteksi dari COCO dataset
category_index = {
    1: {'id': 1, 'name': 'person'},
    17: {'id': 17, 'name': 'cat'},
    18: {'id': 18, 'name': 'dog'},
    19: {'id': 19, 'name': 'horse'},
    20: {'id': 20, 'name': 'sheep'},
    21: {'id': 21, 'name': 'cow'},
    22: {'id': 22, 'name': 'elephant'},
    23: {'id': 23, 'name': 'bear'},
    24: {'id': 24, 'name': 'zebra'},
    25: {'id': 25, 'name': 'giraffe'},
}

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame ke tensor dengan dtype uint8 (penting!)
    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)

    # Jalankan deteksi
    detections = detect_fn(input_tensor)

    # Ambil hasil deteksi
    num_detections = int(detections.pop('num_detections'))
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()
    detection_boxes = detections['detection_boxes'][0].numpy()

    h, w, _ = frame.shape

    # Loop tiap deteksi dan gambar bounding box
    for i in range(num_detections):
        if detection_scores[i] > 0.5:
            class_id = detection_classes[i]
            if class_id in category_index:
                box = detection_boxes[i]
                y1, x1, y2, x2 = box
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

                label = category_index[class_id]['name']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan frame hasil deteksi
    cv2.imshow("Animal Detector", frame)

    # Tekan ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
