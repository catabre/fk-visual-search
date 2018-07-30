import glob
import json
import random
import csv
import os

__author__ = 'ananya.h'

def sample(verticals, output_path, train=True):
    base_dir = "/home/300002291/fk-visual-search/data"
    meta_dir = os.path.join(base_dir, "meta", "json")
    base_image_dir = os.path.join(base_dir, "structured_images")
    number_of_n = 100
    prefix = "train" if train else "test"
    for vertical in verticals:
        filename = prefix + "_pairs_" + vertical + ".json"
        retrieval_path = os.path.join(meta_dir, "retrieval_" + vertical + ".json")
        image_dir = os.path.join(base_image_dir, vertical )
        query_dir = os.path.join(base_image_dir, "wtbi_" + vertical + "_query_crop")
        with open(os.path.join(meta_dir, filename)) as jsonFile:
            pairs = json.load(jsonFile)
        photo_to_product_map = {}
        with open(retrieval_path) as jsonFile:
            data = json.load(jsonFile)
        for info in data:
            photo_to_product_map[info["photo"]] = info["product"]
        product_to_photo_map = {}
        for photo in photo_to_product_map:
            product = photo_to_product_map[photo]
            if product not in product_to_photo_map:
                product_to_photo_map[product] = set()
            product_to_photo_map[product].add(photo)
        universe = [int(os.path.splitext(os.path.basename(x))[0]) for x in
                    glob.glob(image_dir + "/*.jpg")]
        for pair in pairs:
            photo = pair["photo"]
            product = pair["product"]
            p_s = []
            for i in product_to_photo_map[product]:
                p_s.append(i)
            triplets = []
            for p in p_s:
                for j in range(number_of_n):
                    q_id = str(photo)
                    p_id = str(p)
                    n_index = random.randint(0, len(universe) - 1)
                    n = universe[n_index]
                    if n not in p_s and n!=photo:
                        n_id = str(n)
                        triplets.append([q_id, p_id, n_id, vertical])
                with open(output_path, "a") as csvFile:
                    writer = csv.writer(csvFile)
                    triplets = [[os.path.join(query_dir, x[0] + ".jpg"), os.path.join(image_dir, x[1] + ".jpg"),
                             os.path.join(image_dir, x[2] + ".jpg"), x[3]] for x in triplets]
                    if os.path.exists(triplets[0][0]) and os.path.exists(triplets[0][1]) and os.path.exists(triplets[0][2]):
                         print("Successfullly Saved the triplet")
                         writer.writerows(triplets)
                    else: 
                        print("Error Image Not Found:" +  triplets[0][0] + " :: " + triplets[0][1] + " :: " + triplets[0][2] )
                    triplets = []


if __name__ == '__main__':
     sample(verticals=["skirts"], output_path="/home/300002291/fk-visual-search/data/TripletCSV/Test.csv", train=False)
     sample(verticals=["skirts"], output_path="/home/300002291/fk-visual-search/data/TripletCSV/Train.csv", train=True)
