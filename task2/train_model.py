from ultralytics import YOLO


if __name__ == '__main__':
    
    model = YOLO('yolov8n.pt')
    
    model = YOLO('runs/detect/train/weights/best.pt') 

    print("模型已加载，准备开始训练...")
    
    results = model.train(
        data='VisDrone.yaml',   #
        epochs=20,             
        imgsz=640,              
        batch=8,                
        workers=4,              
     
        project='CV_Midterm_Runs',  
        name='finetune_20_epochs'   
    )

    print(" 训练彻底完成！")
    print(" 新的模型权重 (best.pt) 保存在了: CV_Midterm_Runs/finetune_20_epochs/weights/ 目录下")