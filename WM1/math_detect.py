import os
import cv2
from ultralytics import YOLO

def process_math(results,image_shape):
    result = results[0]
    figure_list = []
    # text_list = []
    h, w = image_shape[:2]
    bboxes = result.boxes
    
    for bbox in bboxes:
        xywh = bbox.xywh
        xyxy = bbox.xyxy.cpu().numpy()[0]
        conf = bbox.conf.cpu().numpy()[0]
        box_w = int(xywh[0][2])
        box_h = int(xywh[0][3])
        if box_h < 1:
            continue
        bbox_info = [int(xyxy[0]), int(xyxy[1]),int(xyxy[2]),int(xyxy[3]),conf]
        if box_w >= box_h and box_w >= w * 0.02 and box_w/box_h < 12 and h * 0.01 <= box_h < h*0.4:
            figure_list.append(bbox_info)
            continue
        if box_w < box_h and box_h >= h * 0.02 and box_w/box_h < 12 and w * 0.01 <= box_w < w*0.4:
            figure_list.append(bbox_info)   
            continue
    return figure_list

if __name__ == "__main__":
    math_model_path = "./best.pt"
    math_model = YOLO(math_model_path)
    img_path = "./test_image"
    img_files = os.listdir(img_path)
    save_path = "./test_1110_math"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for img_file in img_files:
        # print(img_file)
        file_img = os.path.join(img_path, img_file)
        image = cv2.imread(file_img)
        results = math_model.predict(image, save=False, imgsz=640, conf=0.2)
        
        result = process_math(results, image.shape)
        if result is not None:
            for figure in result:
                xmin = int(figure[0])
                ymin = int(figure[1])
                xmax = int(figure[2])
                ymax = int(figure[3])
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),2)
        cv2.imwrite(os.path.join(save_path, img_file), image)
        # for result in results:

