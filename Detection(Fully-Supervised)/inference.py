
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import cv2
class_list = {
    "0": "Madonna del Cardillo",
    "1":"Tabernacolo con la Madonna col Bambino",
    "2":"Lastra tombale di Giovanni Cabastida r",
    "3":"Lastra tombale di Giovanni Cabastida f",
    "4":"Piatto fondo",
    "5":"Libro d'Ore miniato",
    "6":"Annunciazione",
    "7":"1-Dettaglio Arcangelo Gabriele superiore",
    "8":"2-Dettaglio Vergine ",
    "9":"3-Dettaglio Arcangelo Gabriele parte inferiore",
    "10":"4-Dettaglio Capitello",
    "11":"5-Dettaglio letto",
    "12":"6-Devoto in basso a dx",
    "13":"7-Dettaglio finestra centrale",
    "14":"8-Dettaglio Sacre Scritture",
    "15":"Altro"
}

mapping_color_for_image={
0: [230, 25, 75],
1: [60, 180, 75],
2: [205, 225, 25],
3: [33, 60, 111],
4: [245, 130, 48],
5: [145, 30, 180],
6: [0, 100, 240],
7: [240, 50, 230],
8: [210, 245, 60],
9: [250, 190, 212],
10:[0, 128, 128],
11: [220, 190, 255],
12: [170, 110, 40],
13: [255, 250, 200],
14: [128, 0, 0],
15: [100,100,100],
}

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x-2,y-2), (x + text_w+2, y + text_h+2), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + int(font_scale) - 1), font, font_scale, text_color, font_thickness)

    return text_size


def main():
    print("INFERENCE_")
    cfg = get_cfg()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file("./detectron2/tools/output/config.yaml")
    
    cfg.MODEL.WEIGHTS = ("./output/model_final.pth")
      # set threshold for this model
    # Create predictor
    file_name = "13_6_tour_1819"
    predictor = DefaultPredictor(cfg)
    im = cv2.imread("./Dataset/all_frames/"+file_name+".jpg")
    # Make prediction

    outputs = predictor(im)

    #outputs["instances"].set('pred_classes', class_list[str(outputs["instances"].pred_classes.item())])
    
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    index=0
    alredy_done =[]
    for box in outputs["instances"].pred_boxes.to('cpu'):
        print((outputs["instances"].scores))
        if(float(outputs["instances"].scores[index].item()) >0.7):
            if(int(outputs["instances"].pred_classes[index].item()) not in alredy_done):
                print(box)
                bb_class =str(outputs["instances"].pred_classes[index].item())
                color = []
                color = [x / 255 for x in mapping_color_for_image[int(bb_class)]]
                a=box[0]
                b=box[1]
                c=box[2]
                d= box[3]
               
                im =cv2.rectangle(im, (int(a), int(b)), (int(c), int(d)),mapping_color_for_image[int(bb_class)], 3)
                draw_text(im,class_list[str(bb_class)],font_scale=0.6,pos=(int(a),int(b+15)),text_color=(0, 0, 0),text_color_bg=mapping_color_for_image[int(bb_class)])
                
                v.draw_text(str(class_list[bb_class]), tuple(box[:2].numpy()),color = color)
                alredy_done.append(int(outputs["instances"].pred_classes[index].item()))
            index+=1
            
    v = v.get_output()
    img =  v.get_image()[:, :, ::-1]
    #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  
    cv2.imwrite("./test/"+file_name+".png", im) 


if __name__ == "__main__":
    main()