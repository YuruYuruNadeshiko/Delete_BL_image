from pywinauto import Desktop
import mss, cv2
import numpy as np
from PIL import Image

# 1. 대화목록 영역 좌표 가져오기
win = Desktop(backend="uia").window(title="은빈")
win.set_focus()
r = win.child_window(auto_id="100", control_type="Pane").rectangle()
region = {"left": r.left, "top": r.top, "width": r.width(), "height": r.height()}
print("캡처 영역:", region)

# 2. 캡처
with mss.MSS() as sct:
    shot = sct.grab(region)
    img = np.array(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"))

# 3. 사진 후보 검출 (큰 사각형 윤곽 찾기)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 30, 100)
edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # 너무 작거나 가느다란 것(텍스트)은 제외 — 사진다운 크기만
    if w > 80 and h > 80 and 0.5 < w/h < 2.0:
        boxes.append((x, y, w, h))

print(f"검출된 사진 후보: {len(boxes)}개")
with mss.MSS() as sct:
    shot = sct.grab(region)
    img = np.array(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"))

Image.fromarray(img).save("raw.png")   # <-- 이 줄 추가



boxes = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # 너무 작거나 가느다란 것(텍스트)은 제외 — 사진다운 크기만
    if w > 80 and h > 80 and 0.5 < w/h < 2.0:
        boxes.append((x, y, w, h))

print(f"검출된 사진 후보: {len(boxes)}개")

# 4. 빨간 박스 그려서 저장
out = img.copy()
for (x, y, w, h) in boxes:
    cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 3)
Image.fromarray(out).save("detected.png")
print("저장 완료: detected.png (빨간 박스 확인)")
