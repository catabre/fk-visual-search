import glob
import json
import os
import cv2
import traceback

__author__ = 'ananya.h'

base_dir = "/home/300002291/fk-visual-search/data"
meta_dir = os.path.join(base_dir, "meta", "json")
structured_dir = os.path.join(base_dir, "structured_images")
query_files = glob.glob(meta_dir + "/*_pairs_*.json")

for path in query_files:
    vertical = path.split("_")[-1].split(".")[0]
    print("Vertical:"+vertical)
    wtbi_crop_dir = os.path.join(structured_dir, "wtbi_"+vertical+"_query_crop")
    if not os.path.exists(wtbi_crop_dir):
        os.mkdir(wtbi_crop_dir)
    query_dir = os.path.join(structured_dir, vertical+"_query")
    print("Processing path %s"%(path))
    with open(path) as jsonFile:
        pairs = json.load(jsonFile)
    for pair in pairs:
        query_id = pair["photo"]
        bbox = pair["bbox"]
        query_path = os.path.join(query_dir, str(query_id)+".jpg")
        print("QueryPath: " + str(query_path))
        if not os.path.exists(query_path):
            continue
        try:
            img = cv2.imread(query_path, cv2.IMREAD_COLOR)
            x, w, y, h = bbox["left"], bbox["width"], bbox["top"], bbox["height"]
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            print("X = " + str(x) + "Y = " + str(y) + "W  = " + str(w)  + "H = " + str(h))     
            crop_img = img[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(wtbi_crop_dir, str(query_id)+".jpg"), crop_img)
        except: 
            print("Exception occured for the Query path:" + str(query_path))
            traceback.print_exc()
