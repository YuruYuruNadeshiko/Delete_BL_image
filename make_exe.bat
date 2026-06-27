@echo off
title Kakao AI Auto Deleter EXE Builder
cd /d "%~dp0"

echo [1/3] PyInstaller 빌더 강제 갱신 중...
"C:\Users\asdf\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pip install --upgrade pip
"C:\Users\asdf\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pip install pyinstaller

echo.
echo [2/3] EXE 실행 파일 패키징 중 (대형 AI 라이브러리 내장으로 수 분이 소요됩니다)...
"C:\Users\asdf\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m PyInstaller --noconfirm --onedir --console --clean --copy-metadata torch --copy-metadata tqdm --collect-all torch --collect-all clip auto_delete_BL.py

echo.
echo [3/3] 빌드 완료! dist 폴더 안으로 reference 폴더 복사 중...
if exist "reference" (
    xcopy "reference" "dist\auto_deleter_mouse\reference\" /E /I /Y
    echo ==> reference 폴더 자동 싱크 완료!
) else (
    echo ==> [경고] reference 폴더를 찾지 못했습니다. 수동으로 dist 내부에 넣어주셔야 합니다.
)

echo.
echo ============================================================
echo 모든 변환 작업이 마무리되었습니다!
echo dist\auto_deleter_mouse 폴더 내부를 확인해 주세요.
echo ============================================================
pause