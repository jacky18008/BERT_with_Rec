import argparse
from collections import defaultdict
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_path')
parser.add_argument('--test_ratio', default=0.2, type=int)
parser.add_argument('--output_path')
args = parser.parse_args()

ui_dict = defaultdict(list)
with open(args.data_path, "r") as fp:
    for line in tqdm(fp):
        line = line.strip("\n")
        user_id, item_id, weight = line.split("\t")
        ui_dict[user_id].append(item_id)
    
with open(args.output_path+".train", "w") as fp_train:
    with open(args.output_path+".test", "w") as fp_test:
        for user_id in tqdm(ui_dict.keys()):
            item_ids = ui_dict[user_id]
            random.shuffle(item_ids)
            train_index = int(len(item_ids)*(1-args.test_ratio))
            train_item_id, test_item_id = item_ids[:train_index], item_ids[train_index:]
            for item_id in train_item_id:
                fp_train.write(f"{user_id}\t{item_id}\t{1}\n")
            for item_id in test_item_id:
                fp_test.write(f"{user_id}\t{item_id}\t{1}\n")