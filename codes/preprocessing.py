import argparse
from collections import defaultdict
from tqdm import tqdm
import json
import sys
# sys.setrecursionlimit(10**7) # recursion limit for k core

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_path')
parser.add_argument('--interaction_path')
parser.add_argument('--text_path')
parser.add_argument('--k', default=10, type=int)
args = parser.parse_args()

core_dict = defaultdict(list) # dict to calculate k-core
core_degree = defaultdict(int)

with open(args.data_path) as fp:
    for line in tqdm(fp):
        line = line.strip("\n")
        review = json.loads(line)
        user_id = review["reviewerID"]
        item_id = review["asin"]
        core_dict[user_id].append(item_id)
        core_dict[item_id].append(user_id)
        core_degree[user_id] += 1
        core_degree[item_id] += 1
total_key = len(core_dict.keys())

# k-core removal
visited = set()
def traverse(core_dict, k):
    removed = []
    to_traverse = list(core_dict.keys())
    for key in tqdm(to_traverse):
        if core_degree[key] < k:
            removed.append(key)
            core_degree.pop(key)
            try:
                for key_ in core_dict[key]: 
                    core_degree[key_] -= 1
            except:
                continue
            core_dict.pop(key)
    return removed, core_dict

removed = [-1]
while removed:
    removed, core_dict = traverse(core_dict, args.k)
    # print("dict len: ", len(core_dict.keys()))
    print("remove len: ", len(removed))
print("final len: ", len(core_dict.keys()))

# go through data again and assign user/item id
user_remap_id = {}
item_remap_id = {}
item_texts = defaultdict(list)
user_counter = 0
item_counter = 0
with open(args.data_path) as fp_in:
    with open(args.interaction_path, "w") as fp_out:
        for line in tqdm(fp_in):
            line = line.strip("\n")
            review = json.loads(line)
            user_id = review["reviewerID"]
            item_id = review["asin"]
            if core_degree[user_id] >= args.k and core_degree[item_id] >= args.k:
                try:
                    text = review["summary"] # collect summary here
                except:
                    text = ""
                if user_id not in user_remap_id:
                    user_remap_id[user_id] = user_counter
                    user_counter += 1
                if item_id not in item_remap_id:
                    item_remap_id[item_id] = item_counter
                    item_counter += 1
                if text != "":
                    item_texts[item_remap_id[item_id]].append(text)
                fp_out.write(f"{user_remap_id[user_id]}\t{item_remap_id[item_id]}\t1\n")

# squeeze item texts
with open(args.text_path, "w") as fp:
    for item_id, text in item_texts.items():
        fp.write(f"{item_id}\t{', '.join(text)}\n")

    

    

    
    