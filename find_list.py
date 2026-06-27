from pywinauto import Desktop

win = Desktop(backend="uia").window(title="# 여기다 이름 입력")
lst = win.child_window(auto_id="100", control_type="Pane")
rect = lst.rectangle()
print("대화목록 영역:", rect)
print(f"left={rect.left} top={rect.top} width={rect.width()} height={rect.height()}")
