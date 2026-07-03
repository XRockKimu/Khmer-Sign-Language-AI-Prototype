print("Program started")

import cv2
from ai_inference.keypoint_extractor import HandKeypointExtractor

print("Imports successful")

extractor = HandKeypointExtractor()
print("Extractor created")

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())

while True:
    ret, frame = cap.read()
    print("Frame read:", ret)

    if not ret:
        break

    data = extractor.extract(frame)

    print(
        data["left_hand"].shape,
        data["right_hand"].shape
    )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
extractor.close()
cv2.destroyAllWindows()