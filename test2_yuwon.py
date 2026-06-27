import clip, torch, os, glob
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def embed(path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        v = model.encode_image(img)
    return (v / v.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

# 기준 이미지(유원불변) 평균 벡터
exts = ("*.jpg","*.jpeg","*.png","*.webp","*.bmp","*.jfif")
ref_paths = []
for e in exts:
    ref_paths += glob.glob(f"reference/{e}")
ref_vecs = [embed(p) for p in ref_paths]
ref_mean = np.mean(ref_vecs, axis=0)
ref_mean /= np.linalg.norm(ref_mean)
print(f"기준 이미지 {len(ref_vecs)}장 로드됨")

# 폴더별 유사도 출력
def show_sim(folder):
    paths = []
    for e in exts:
        paths += glob.glob(f"{folder}/{e}")
    print(f"\n--- {folder} ({len(paths)}장) ---")
    for p in paths:
        sim = float(np.dot(embed(p), ref_mean))
        print(f"  {sim:.3f}  {p}")

show_sim("test_yuwon")
show_sim("test_others")
