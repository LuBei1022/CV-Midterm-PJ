import cv2
import os
import glob
from ultralytics import YOLO

model_path = r"runs\detect\CV_Midterm_Runs\finetune_20_epochs-2\weights\best.pt"
model = YOLO(model_path)

input_folder = r'VisDrone2019-MOT-test-dev\VisDrone2019-MOT-test-dev\sequences\uav0000355_00001_v'
img_paths = sorted(glob.glob(os.path.join(input_folder, '*.jpg')))

if not img_paths:
    print(f"未能找到图片，请仔细检查路径是否正确: {input_folder}")
    exit()

first_frame = cv2.imread(img_paths[0])
h, w = first_frame.shape[:2]
out_video = cv2.VideoWriter('output_final_counted1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

LINE_START = (0, int(h * 0.6))      
LINE_END = (w, int(h * 0.6))        

track_history = {}  
crossed_ids = set() 
total_count = 0     

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
# ================================================================

print(f"找到 {len(img_paths)} 张图片，开始追踪与越线计数...")

for img_path in img_paths:
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.35, iou=0.5, verbose=False)
    annotated_frame = results[0].plot()
    cv2.line(annotated_frame, LINE_START, LINE_END, (0, 0, 255), 3)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  
        track_ids = results[0].boxes.id.int().cpu().numpy()  
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if track_id in track_history:
                prev_center = track_history[track_id] 
                current_center = (cx, cy)            

                if intersect(prev_center, current_center, LINE_START, LINE_END):
                    if track_id not in crossed_ids:
                        crossed_ids.add(track_id) 
                        total_count += 1
                        print(f"-> 检测到 目标 ID={track_id} 越线！当前总流量: {total_count}")


            track_history[track_id] = (cx, cy)

            cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.putText(annotated_frame, f"Traffic Count: {total_count}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

    out_video.write(annotated_frame)


out_video.release()
print(f"\n总计越线目标数: {total_count}")
print("最终视频已保存为本目录下的: output_final_counted1.mp4")