import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import sys
import io
import os
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as messagebox
from cellpose import models

# 🌟 PyTorch 및 이미지 처리 라이브러리 추가
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

# ===============================
# 0. 파일 선택 및 초기 설정
# ===============================
root = tk.Tk()
root.withdraw() 
root.attributes('-topmost', True)

def resource_path(relative_path):
    """ PyInstaller로 빌드된 환경(임시 폴더)과 일반 파이썬 환경을 모두 지원하는 경로 추적 함수 """
    try:
        # PyInstaller가 실행될 때 데이터를 풀어놓는 임시 폴더 경로 (_MEIPASS)
        base_path = sys._MEIPASS
    except Exception:
        # 일반 파이썬 스크립트로 실행할 때의 현재 폴더 경로
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
# 🌟 프로그램 창과 작업 표시줄에 아이콘 적용하기
try:
    root.iconbitmap(resource_path("malaria.ico"))
except Exception:
    pass # 혹시 아이콘 파일이 없어도 에러 나지 않게 패스

image_path = filedialog.askopenfilename(
    title="분석할 현미경 이미지 선택",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")]
)

if not image_path:
    sys.exit()



# 🌟 한글 경로 지원을 위한 우회 읽기 방식 (numpy + imdecode)
img_array = np.fromfile(image_path, np.uint8)
img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

if img_bgr is None:
    messagebox.showerror("파일 읽기 오류", "이미지를 읽을 수 없습니다.\n파일이 손상되었거나 지원하지 않는 이미지 형식일 수 있습니다.")
    sys.exit()

# ===============================
# ⏳ 로딩 팝업창 띄우기
# ===============================
loading_popup = tk.Toplevel(root)
loading_popup.title("분석 중...")
loading_popup.geometry("350x120")
loading_popup.attributes('-topmost', True)

window_width, window_height = 350, 120
screen_width = loading_popup.winfo_screenwidth()
screen_height = loading_popup.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
loading_popup.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

tk.Label(loading_popup, text="AI 모델을 불러오고 세포를 분석하는 중입니다.\n컴퓨터 사양에 따라 10~30초 정도 소요될 수 있습니다...\n\n잠시만 기다려주세요.", font=("맑은 고딕", 10)).pack(expand=True)
root.update() 

# ===============================
# 1. 전처리 및 Cellpose 알고리즘 적용
# ===============================
orig_bgr = img_bgr.copy()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

model = models.Cellpose(gpu=False, model_type='cyto3') 
masks, flows, styles, diams = model.eval(img_rgb, diameter=None, channels=[0, 0])
total_cells = np.max(masks)

# ===============================
# 1.5 가장자리 잘린 세포(Edge Cells) 필터링 (Margin 30px 적용)
# ===============================
height, width = masks.shape
cell_areas = np.bincount(masks.flatten())

MARGIN = 30  
edge_cells = set(masks[:MARGIN, :].flatten()) | set(masks[-MARGIN:, :].flatten()) | \
             set(masks[:, :MARGIN].flatten()) | set(masks[:, -MARGIN:].flatten())
edge_cells.discard(0)

inner_cells_areas = [cell_areas[cid] for cid in range(1, total_cells + 1) if cid not in edge_cells]
standard_area = np.median(inner_cells_areas) if inner_cells_areas else np.median(cell_areas[1:])

valid_cells = []
for cell_id in range(1, total_cells + 1):
    if cell_id in edge_cells and cell_areas[cell_id] < (standard_area * 0.9):
        continue  
    if cell_areas[cell_id] < (standard_area * 0.4):
        continue
    valid_cells.append(cell_id)

# ===============================
# 2. PyTorch 딥러닝 모델로 감염 상태 판별
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 아키텍처 불러오기 (2진 분류: Infected, Uninfected)
class_names = ['Infected', 'Uninfected']
num_classes = len(class_names)
resnet_model = tv_models.resnet50(weights=None)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

# 🌟 가중치 파일 로드 (.exe 내부 임시 폴더 경로 추적 기능 추가)


model_path = resource_path("resnet50_malaria.pth")
if not os.path.exists(model_path):
    loading_popup.destroy()
    messagebox.showerror("모델 오류", f"'{model_path}' 파일을 찾을 수 없습니다.\n파이썬 코드와 같은 폴더에 있는지 확인해주세요.")
    sys.exit()

resnet_model.load_state_dict(torch.load(model_path, map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_cells_set = set(valid_cells)
infected_cells_set = set()
cell_contours = {}

# 🌟 현미경 슬라이드의 자연스러운 평균 배경색 계산
background_mask = (masks == 0).astype(np.uint8)
bg_mean_color = cv2.mean(img_rgb, mask=background_mask)[:3]
bg_mean_color = np.array(bg_mean_color, dtype=np.uint8)

margin = 5 # 데이터 추출기에서 사용했던 마진과 동일하게 유지

for cell_id in range(1, total_cells + 1):
    cell_mask = (masks == cell_id).astype(np.uint8)
    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_contours[cell_id] = contours
    
    # 제외된 세포는 딥러닝 분석 안 함 (속도 향상)
    if cell_id not in valid_cells_set:
        continue
        
    y_idx, x_idx = np.where(cell_mask == 1)
    if len(y_idx) > 0:
        y_min, y_max = np.min(y_idx), np.max(y_idx)
        x_min, x_max = np.min(x_idx), np.max(x_idx)
        
        y_min_m = max(0, y_min - margin)
        y_max_m = min(img_rgb.shape[0], y_max + margin)
        x_min_m = max(0, x_min - margin)
        x_max_m = min(img_rgb.shape[1], x_max + margin)
        
        # 1. 원본 자르기
        roi_rgb = img_rgb[y_min_m:y_max_m, x_min_m:x_max_m].copy()
        
        # 2. 이웃 세포 지우기 로직
        roi_masks_full = masks[y_min_m:y_max_m, x_min_m:x_max_m]
        neighbor_idx = (roi_masks_full != cell_id) & (roi_masks_full != 0)
        roi_rgb[neighbor_idx] = bg_mean_color
        
        if roi_rgb.shape[0] < 20 or roi_rgb.shape[1] < 20: 
            continue
            
        # 3. 모델 추론
        pil_img = Image.fromarray(roi_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = resnet_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            prob_infected = probabilities[0].item() * 100
            
            # 🌟 감염 확률이 80% 이상일 때만 감염으로 판정 (필요시 조절)
            if prob_infected > 80.0:
                infected_cells_set.add(cell_id)

# ===============================
# 3. 분석 완료 -> 로딩창 닫고 결과 시각화
# ===============================
loading_popup.destroy() 

from matplotlib.widgets import Button, TextBox

fig, ax = plt.subplots(figsize=(10, 9))
plt.subplots_adjust(bottom=0.2) 

# 🌟 내가 따로 더하거나 뺄 '수동 입력(Offset)' 변수
manual_inf_adj = 0
manual_valid_adj = 0

def get_base_counts():
    base_valid = len(valid_cells_set)
    base_inf = len(infected_cells_set.intersection(valid_cells_set))
    return base_valid, base_inf

def update_plot():
    output = orig_bgr.copy()
    
    for cell_id in range(1, total_cells + 1):
        contours = cell_contours.get(cell_id, [])
        if not contours: continue
            
        if cell_id in valid_cells_set:
            if cell_id in infected_cells_set:
                color = (0, 0, 255) # 빨간색
            else:
                color = (0, 255, 0) # 초록색
        else:
            color = (128, 128, 128) # 회색
            
        cv2.drawContours(output, contours, -1, color, 2)
        
    # 🌟 1. 화면에서 인식/클릭된 기본 개수
    base_valid, base_inf = get_base_counts()
    
    # 🌟 2. 최종 계산값 = 기본 개수 + 수동 입력 개수
    tot_valid = max(0, base_valid + manual_valid_adj)
    tot_inf = max(0, base_inf + manual_inf_adj)
    
    parasitemia = (tot_inf / tot_valid * 100) if tot_valid > 0 else 0
    
    ax.clear()
    ax.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    
    # 🌟 3. 타이틀에 계산식 명확히 표시 (예: Infected: 8 + (2) = 10)
    title_text = (f"Malaria AI (Parasitemia: {parasitemia:.2f}%)\n"
                  f"Valid: {base_valid} + ({manual_valid_adj:+}) = {tot_valid}  |  "
                  f"Infected: {base_inf} + ({manual_inf_adj:+}) = {tot_inf}\n"
                  f"[L-Click]: Toggle Infected  |  [R-Click]: Toggle Valid")
    ax.set_title(title_text, fontsize=12, pad=10)
    ax.axis("off")
    
    # 텍스트 박스에 수동 입력 숫자 업데이트 (무한 루프 방지 처리)
    if txt_inf.text != str(manual_inf_adj): txt_inf.set_val(str(manual_inf_adj))
    if txt_val.text != str(manual_valid_adj): txt_val.set_val(str(manual_valid_adj))
    
    fig.canvas.draw_idle()

def onclick(event):
    if event.inaxes != ax: return 
    if event.xdata is None or event.ydata is None: return
        
    x, y = int(event.xdata), int(event.ydata)
    cell_id = masks[y, x] 
    if cell_id == 0: return 
        
    if event.button == 1: # 좌클릭
        if cell_id not in valid_cells_set:
            valid_cells_set.add(cell_id)
        if cell_id in infected_cells_set:
            infected_cells_set.remove(cell_id) 
        else:
            infected_cells_set.add(cell_id)    
            
    elif event.button == 3: # 우클릭
        if cell_id in valid_cells_set:
            valid_cells_set.remove(cell_id) 
        else:
            valid_cells_set.add(cell_id)    
            
    update_plot()

fig.canvas.mpl_connect('button_press_event', onclick)

# ===============================
# 🌟 하단 패널: "추가/제외할 숫자(Offset)"만 조작하는 UI
# ===============================
# Infected 수동 조작부
ax_inf_label = plt.axes([0.1, 0.05, 0.15, 0.05])
ax_inf_label.axis('off')
ax_inf_label.text(0.5, 0.5, 'Manual Inf ± :', ha='center', va='center', fontsize=11, fontweight='bold')

ax_inf_minus = plt.axes([0.25, 0.05, 0.04, 0.05])
ax_inf_text  = plt.axes([0.30, 0.05, 0.06, 0.05])
ax_inf_plus  = plt.axes([0.37, 0.05, 0.04, 0.05])

# Total Valid 수동 조작부
ax_val_label = plt.axes([0.55, 0.05, 0.15, 0.05])
ax_val_label.axis('off')
ax_val_label.text(0.5, 0.5, 'Manual Valid ± :', ha='center', va='center', fontsize=11, fontweight='bold')

ax_val_minus = plt.axes([0.70, 0.05, 0.04, 0.05])
ax_val_text  = plt.axes([0.75, 0.05, 0.06, 0.05])
ax_val_plus  = plt.axes([0.82, 0.05, 0.04, 0.05])

# 위젯 장착
btn_inf_minus = Button(ax_inf_minus, '-')
btn_inf_plus  = Button(ax_inf_plus, '+')
txt_inf       = TextBox(ax_inf_text, '', textalignment='center')

btn_val_minus = Button(ax_val_minus, '-')
btn_val_plus  = Button(ax_val_plus, '+')
txt_val       = TextBox(ax_val_text, '', textalignment='center')

# 위젯 기능 정의 (이제 기존 개수와 상관없이 순수하게 '수동 조작값'만 +- 합니다)
def inf_minus(event): global manual_inf_adj; manual_inf_adj -= 1; update_plot()
def inf_plus(event):  global manual_inf_adj; manual_inf_adj += 1; update_plot()
def submit_inf(text):
    global manual_inf_adj
    try: manual_inf_adj = int(text); update_plot()
    except ValueError: update_plot()

def val_minus(event): global manual_valid_adj; manual_valid_adj -= 1; update_plot()
def val_plus(event):  global manual_valid_adj; manual_valid_adj += 1; update_plot()
def submit_val(text):
    global manual_valid_adj
    try: manual_valid_adj = int(text); update_plot()
    except ValueError: update_plot()

# 기능 연결
btn_inf_minus.on_clicked(inf_minus)
btn_inf_plus.on_clicked(inf_plus)
txt_inf.on_submit(submit_inf)

btn_val_minus.on_clicked(val_minus)
btn_val_plus.on_clicked(val_plus)
txt_val.on_submit(submit_val)

# 초기 렌더링
update_plot()
plt.show()