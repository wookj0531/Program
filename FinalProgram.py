import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import sys
import io
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as messagebox
from cellpose import models

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

# ===============================
# 0. 파일 선택 및 초기 설정
# ===============================
# GUI 창을 띄우기 위한 기본 설정
root = tk.Tk()
root.withdraw() # 메인 창 숨기기
root.attributes('-topmost', True)

# 파일 선택 창 띄우기
image_path = filedialog.askopenfilename(
    title="분석할 현미경 이미지 선택",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")]
)

# 파일 선택 취소 시 조용히 종료
if not image_path:
    sys.exit()

# 이미지 읽기 및 에러 처리 (팝업창)
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    messagebox.showerror("파일 읽기 오류", "이미지를 읽을 수 없습니다.\n경로에 한글이 포함되어 있거나 손상된 파일일 수 있습니다.")
    sys.exit()

# ===============================
# ⏳ 로딩 팝업창 띄우기
# ===============================
loading_popup = tk.Toplevel(root)
loading_popup.title("분석 중...")
loading_popup.geometry("350x120")
loading_popup.attributes('-topmost', True)

# 화면 중앙에 로딩창 띄우기 위한 위치 계산
window_width, window_height = 350, 120
screen_width = loading_popup.winfo_screenwidth()
screen_height = loading_popup.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
loading_popup.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

tk.Label(loading_popup, text="AI 모델을 불러오고 세포를 분석하는 중입니다.\n컴퓨터 사양에 따라 10~30초 정도 소요될 수 있습니다...\n\n잠시만 기다려주세요.", font=("맑은 고딕", 10)).pack(expand=True)
root.update() # 화면 강제 새로고침하여 팝업창을 즉시 렌더링

# ===============================
# 1. 전처리 및 Cellpose 알고리즘 적용
# ===============================
orig_bgr = img_bgr.copy()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

# Cellpose 모델 로드 및 분할 (로딩 팝업이 떠 있는 동안 백그라운드에서 실행됨)
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
# 2. 초기 감염 상태 판별 및 데이터 세팅 (사전 계산)
# ===============================
PARASITE_THR = 10
lower_purple = np.array([120, 100, 50])
upper_purple = np.array([170, 255, 255])

valid_cells_set = set(valid_cells)
infected_cells_set = set()
cell_contours = {}

for cell_id in range(1, total_cells + 1):
    cell_mask = (masks == cell_id).astype(np.uint8)
    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_contours[cell_id] = contours
    
    y_idx, x_idx = np.where(cell_mask == 1)
    if len(y_idx) > 0:
        y_min, y_max = np.min(y_idx), np.max(y_idx)
        x_min, x_max = np.min(x_idx), np.max(x_idx)
        roi_hsv = hsv_img[y_min:y_max+1, x_min:x_max+1]
        roi_mask = cell_mask[y_min:y_max+1, x_min:x_max+1]
        
        parasite_mask = cv2.inRange(roi_hsv, lower_purple, upper_purple)
        parasite_in_cell = cv2.bitwise_and(parasite_mask, parasite_mask, mask=roi_mask)
        
        if cv2.countNonZero(parasite_in_cell) > PARASITE_THR:
            infected_cells_set.add(cell_id)

# ===============================
# 3. 분석 완료 -> 로딩창 닫고 결과 시각화
# ===============================
loading_popup.destroy() # 로딩이 끝났으므로 Tkinter 창을 완전히 닫습니다.

fig, ax = plt.subplots(figsize=(10, 10))

def update_plot():
    output = orig_bgr.copy()
    
    for cell_id in range(1, total_cells + 1):
        contours = cell_contours.get(cell_id, [])
        if not contours: continue
            
        if cell_id in valid_cells_set:
            if cell_id in infected_cells_set:
                color = (0, 0, 255) # 빨간색 (감염)
            else:
                color = (0, 255, 0) # 초록색 (정상)
        else:
            color = (128, 128, 128) # 회색 (계산 제외/가장자리)
            
        cv2.drawContours(output, contours, -1, color, 2)
        
    tot_valid = len(valid_cells_set)
    tot_inf = len(infected_cells_set.intersection(valid_cells_set)) 
    parasitemia = (tot_inf / tot_valid * 100) if tot_valid > 0 else 0
    
    ax.clear()
    ax.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    
    title_text = (f"Malaria Detection (Parasitemia: {parasitemia:.2f}%) | Valid: {tot_valid}, Infected: {tot_inf}\n"
                  f"[Left Click]: Toggle Infected (Red <-> Green)\n"
                  f"[Right Click]: Toggle Valid RBC (Gray <-> Green/Red)")
    ax.set_title(title_text, fontsize=12, pad=10)
    ax.axis("off")
    fig.canvas.draw_idle()

def onclick(event):
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
update_plot()

# matplotlib 창을 띄웁니다. 사용자가 x버튼을 눌러 닫을 때까지 프로그램은 여기서 대기합니다.
plt.show()