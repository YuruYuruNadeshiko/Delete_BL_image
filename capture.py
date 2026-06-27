import pygetwindow as gw
import mss
from PIL import Image

# 1. 카카오톡 채팅방 창 찾기
#    채팅방을 따로 띄우면 창 제목이 '상대방 이름'이 됩니다.
#    먼저 어떤 창들이 있는지 출력해봅니다.
print("=== 현재 열린 창 목록 ===")
for w in gw.getAllTitles():
    if w.strip():
        print(f"  [{w}]")

# 2. 캡처할 창 제목 (위 목록에서 카톡 채팅방 제목을 골라 넣으세요)
TARGET = "#여기다 이릅 입력"   # <- 여기를 실제 채팅방 제목으로 바꾸기

wins = gw.getWindowsWithTitle(TARGET)
if not wins:
    print(f"\n'{TARGET}' 창을 찾지 못했습니다. 위 목록에서 제목을 골라 TARGET에 넣으세요.")
else:
    win = wins[0]
    win.activate()   # 창을 앞으로 가져오기
    print(f"\n찾음: {win.title}  위치=({win.left},{win.top})  크기=({win.width}x{win.height})")

    # 3. 그 창 영역만 캡처
    region = {"left": win.left, "top": win.top,
              "width": win.width, "height": win.height}
    with mss.mss() as sct:
        shot = sct.grab(region)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        img.save("kakao_capture.png")
    print("저장 완료: kakao_capture.png")
