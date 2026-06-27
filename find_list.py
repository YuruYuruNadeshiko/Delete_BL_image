from pywinauto import Desktop

win = Desktop(backend="uia").window(title="은빈")
lst = win.child_window(auto_id="100", control_type="Pane")
rect = lst.rectangle()
print("대화목록 영역:", rect)
print(f"left={rect.left} top={rect.top} width={rect.width()} height={rect.height()}")
