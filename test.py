import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 판단 기준이 되는 텍스트 후보들 (영어가 정확도 더 좋음)
labels = [
    "a webtoon comic character illustration",  # 웹툰 캐릭터
    "a real photograph of a person",            # 실제 인물 사진
    "a photo of food",                          # 음식
    "a screenshot of text",                     # 텍스트 스크린샷
    "a landscape or scenery photo",             # 풍경
]

def classify(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    print(f"\n[{image_path}]")
    for label, p in sorted(zip(labels, probs), key=lambda x: -x[1]):
        print(f"  {p*100:5.1f}%  {label}")

    # '웹툰 캐릭터'가 1순위면 삭제 후보로 판단
    top = labels[probs.argmax()]
    is_target = (top == labels[0])
    print(f"  => 삭제 후보? {'예' if is_target else '아니오'}")
    return is_target

# 테스트
classify("test1.jpg")
classify("test2.jpg")
