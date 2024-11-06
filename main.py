from ultralytics import YOLO
model = YOLO(r'C:\Users\Rishav\Desktop\me\code\projects\dataset_pbl\YoloV9.pt')
#model.predict(source=r"C:\Users\Rishav\Desktop\me\code\projects\dataset_pbl\istockphoto-467579433-612x612.jpg", imgsz=640, conf=0.30, save=True)
                

model.predict(source=0, imgsz=640, conf=0.30, show=True)
                

