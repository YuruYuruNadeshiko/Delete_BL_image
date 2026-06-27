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


# --------------------------------------------------------
# [배포용 폴더 디펜스] reference 폴더 자동 생성 및 안내
# --------------------------------------------------------
REFERENCE_DIR = "reference"

if not os.path.exists(REFERENCE_DIR):
    os.makedirs(REFERENCE_DIR)
    print("\n" + "!"*60)
    print(f"💡 [{REFERENCE_DIR}] 폴더가 없어서 새로 생성했습니다.")
    print("   생성된 'reference' 폴더 안에 '삭제하고 싶은 타겟 이미지'를 넣은 후")
    print("   아무 키나 누르면 프로그램이 계속 진행됩니다.")
    print("!"*60)
    input("👉 이미지를 넣으셨다면 엔터(Enter)를 눌러주세요...")

exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.jfif")
ref_paths = []
for e in exts:
    ref_paths += glob.glob(f"{REFERENCE_DIR}/{e}")

# 만약 엔터를 눌렀는데도 이미지가 한 장도 없다면 다시 한번 경고
if not ref_paths:
    print("\n" + "="*60)
    print("⚠️ [안내] reference 폴더에 기준 이미지가 존재하지 않습니다.")
    print("   이 상태로 진행하면 '웹툰 만화 캐릭터 이미지'만 판별하여 삭제합니다.")
    print("="*60)
    input("👉 그대로 진행하시려면 엔터(Enter)를 눌러주세요...")
    ref_mean = None
else:
    ref_vecs = [get_image_embedding(Image.open(p)) for p in ref_paths]
    ref_mean = np.mean(ref_vecs, axis=0)
    ref_mean /= np.linalg.norm(ref_mean)
    print(f"✅ 기준 이미지 {len(ref_vecs)}장 기반 유사도 벡터 생성 완료.")


# ==========================================
# 2. 사용자로부터 채팅방 이름 입력받기
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

win_rect = win.rectangle()
win_w = win_rect.width()
win_h = win_rect.height()

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
    # 마우스 제어 시퀀스
    # --------------------------------------------------------
    if is_delete_target:
        print(f"\n🎯 [{idx+1}번 사진] 삭제 타겟 확정. 매크로 제어를 시작합니다.")
        
        screen_x = region["left"] + x + (w // 2)
        screen_y = region["top"] + y + (h // 2)
        
        pyautogui.moveTo(screen_x, screen_y, duration=0.3)
        pyautogui.rightClick()
        time.sleep(0.5)
        
        menu_delete_x = screen_x + 30
        menu_delete_y = screen_y + 195  
        pyautogui.moveTo(menu_delete_x, menu_delete_y, duration=0.3)
        time.sleep(0.5)
        
        submenu_x = menu_delete_x + 120
        submenu_y = menu_delete_y
        pyautogui.moveTo(submenu_x, submenu_y, duration=0.3)
        pyautogui.click()
        time.sleep(0.6)
        
        confirm_btn_x = win_rect.right - 90  
        confirm_btn_y = win_rect.bottom - 25
        
        pyautogui.moveTo(confirm_btn_x, confirm_btn_y, duration=0.3)
        pyautogui.click()
        time.sleep(0.8)
        
        kakao_center_x = win_rect.left + (win_w // 2)
        kakao_center_y = win_rect.top + (win_h // 2)
        
        center_popup_x = kakao_center_x - 60
        center_popup_y = kakao_center_y + 65
        
        pyautogui.moveTo(center_popup_x, center_popup_y, duration=0.3)
        pyautogui.click()
        
        print(f"✅ [{idx+1}번 사진] 나에게서만 삭제 프로세스 완료.")
        time.sleep(1.5)

print("\n🎉 모든 작업이 완료되었습니다.")
input("종료하려면 엔터를 누르세요...")
