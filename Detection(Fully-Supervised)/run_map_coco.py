from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
annType = 'bbox'
prefix = 'instances'
annFile = "../../Dataset/validate_coco.json" #gt file
cocoGT = COCO(annFile)
predsFile = './output/inference/coco_instances_results_clean.json' #predictions file
cocoPreds = cocoGT.loadRes(predsFile)
cocoEval = COCOeval(cocoGT, cocoPreds, annType)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()