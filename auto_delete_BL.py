import os
import time
import glob
import numpy as np
import torch
import clip
from PIL import Image
import cv2
import mss
import pyautogui
from pywinauto import Desktop

# ==========================================
# 1. AI 모델 및 설정 초기화
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

labels = [
    "a webtoon comic character illustration",
    "a real photograph of a person",
    "a photo of food",
    "a screenshot of text",
    "a landscape or scenery photo",
]
text_tokens = clip.tokenize(labels).to(device)

def get_image_embedding(pil_img):
    img_tensor = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        v = model.encode_image(img_tensor)
    vec = (v / v.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
    return vec

exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.jfif")
ref_paths = []
for e in exts:
    ref_paths += glob.glob(f"reference/{e}")

if ref_paths:
    ref_vecs = [get_image_embedding(Image.open(p)) for p in ref_paths]
    ref_mean = np.mean(ref_vecs, axis=0)
    ref_mean /= np.linalg.norm(ref_mean)
    print(f"✅ 기준 이미지 {len(ref_vecs)}장 기반 유사도 벡터 생성 완료.")
else:
    ref_mean = None
    print("⚠️ reference 폴더에 기준 이미지가 없어 유사도 검사는 건너뜁니다.")


# ==========================================
# 2. [배포용 업그레이드] 사용자로부터 채팅방 이름 입력받기
# ==========================================
print("\n" + "="*50)
TARGET_WINDOW = input("🎯 타겟 카카오톡 채팅방 이름을 정확히 입력하세요: ").strip()
print("="*50)

try:
    win = Desktop(backend="uia").window(title=TARGET_WINDOW)
    win.set_focus()
    time.sleep(0.5)
except Exception as e:
    print(f"❌ '{TARGET_WINDOW}' 창을 찾을 수 없습니다. 카톡 창이 켜져 있는지 확인해 주세요.")
    input("종료하려면 엔터를 누르세요...")
    exit()

# 카톡 창 전체의 절대 좌표 구하기
win_rect = win.rectangle()
win_w = win_rect.width()
win_h = win_rect.height()

# 대화 목록 내용 영역(auto_id="100") 좌표 구하기 (사진 검출용)
try:
    lst = win.child_window(auto_id="100", control_type="Pane")
    rect = lst.rectangle()
    region = {"left": rect.left, "top": rect.top, "width": rect.width(), "height": rect.height()}
except Exception as e:
    print("❌ 카톡 내부 대화 영역을 인식하지 못했습니다. 카톡을 기본 테마 크기로 띄워주세요.")
    input("종료하려면 엔터를 누르세요...")
    exit()


# ==========================================
# 3. 화면 캡처 및 사진 박스 검출
# ==========================================
with mss.MSS() as sct:
    shot = sct.grab(region)
    img_bgr = np.array(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"))

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 20, 80)
edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w > 50 and h > 50:
        boxes.append((x, y, w, h))

print(f"🔍 검출된 사진 후보: {len(boxes)}개")


# ==========================================
# 4. 각 박스별 AI 판정 및 4단계 정밀 마우스 클릭 제어
# ==========================================
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
pil_canvas = Image.fromarray(img_rgb)

for idx, (x, y, w, h) in enumerate(boxes):
    cropped_img = pil_canvas.crop((x, y, x + w, y + h))
    
    is_delete_target = False
    sim_score = 0.0
    
    if ref_mean is not None:
        current_vec = get_image_embedding(cropped_img)
        sim_score = float(np.dot(current_vec, ref_mean))
        if sim_score >= 0.81:
            is_delete_target = True

    if not is_delete_target:
        img_tensor = preprocess(cropped_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, _ = model(img_tensor, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        if labels[probs.argmax()] == labels[0]:
            is_delete_target = True

    # --------------------------------------------------------
    # [배포용 해상도 방어] 실측 오프셋 및 상대 비율 기반 연산
    # --------------------------------------------------------
    if is_delete_target:
        print(f"\n🎯 [{idx+1}번 사진] 삭제 타겟 확정. 매크로 제어를 시작합니다.")
        
        # [Step 1] 사진 중심 우클릭
        screen_x = region["left"] + x + (w // 2)
        screen_y = region["top"] + y + (h // 2)
        
        pyautogui.moveTo(screen_x, screen_y, duration=0.3)
        pyautogui.rightClick()
        time.sleep(0.5)
        
        # [Step 2] [삭제] 메뉴 위로 이동 (피드백 수치: 아래로 195픽셀)
        menu_delete_x = screen_x + 30
        menu_delete_y = screen_y + 195  
        pyautogui.moveTo(menu_delete_x, menu_delete_y, duration=0.3)
        time.sleep(0.5)
        
        # [Step 3] 옆에 뜬 [나에게서만 삭제] 메뉴 클릭
        submenu_x = menu_delete_x + 120
        submenu_y = menu_delete_y
        pyautogui.moveTo(submenu_x, submenu_y, duration=0.3)
        pyautogui.click()
        time.sleep(0.6)
        
        # [Step 4] 카톡 창 우측 하단의 노란색 [확인] 버튼 클릭
        # 피드백 주신 가로 -90 위치를 반영하되, 창 크기별 하단 레이아웃 방어를 위해 정밀 조준합니다.
        confirm_btn_x = win_rect.right - 90  
        confirm_btn_y = win_rect.bottom - 25
        
        pyautogui.moveTo(confirm_btn_x, confirm_btn_y, duration=0.3)
        pyautogui.click()
        time.sleep(0.8)
        
        # [Step 5] 채팅방 창 정중앙 기준 화이트 팝업창의 [확인] 버튼 클릭
        # 사용자별 창 크기 변화에 대응하기 위해 카톡 창 정중앙을 먼저 구합니다.
        kakao_center_x = win_rect.left + (win_w // 2)
        kakao_center_y = win_rect.top + (win_h // 2)
        
        # 화이트 팝업 내부의 [확인] 버튼 조준 (중앙선 기준 왼쪽 아래 오프셋 정밀 적용)
        center_popup_x = kakao_center_x - 60
        center_popup_y = kakao_center_y + 65
        
        pyautogui.moveTo(center_popup_x, center_popup_y, duration=0.3)
        pyautogui.click()
        
        print(f"✅ [{idx+1}번 사진] 나에게서만 삭제 프로세스 완료.")
        time.sleep(1.5)