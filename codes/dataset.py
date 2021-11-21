from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np


class Triplet_Dataset(Dataset):
    def __init__(self, uiw_path, itext_path, config):
        self.max_user = config['max_user']
        self.max_item = config['max_item']
        self.load_uiw(uiw_path)
        self.load_text(itext_path)
        print("num users: ", self.n_users)
        print("num items: ", self.m_items)
        print("edges: ", self.num_train)

    def load_uiw(self, uiw_path):
        self.allPos = defaultdict(list)
        self.num_train = 0
        
        with open(uiw_path, 'r') as fp:
            for line in fp:
                line = line.strip("\n")
                user_id, item_id, weight = line.split("\t")
                user_id = int(user_id)
                item_id = int(item_id)
                if user_id > self.max_user or item_id > self.max_item:
                    continue
                self.allPos[user_id].append(item_id)
                self.num_train += 1
        self.n_users = max(self.allPos.keys())+1
        temp = [[] for _ in range(self.n_users)]
        for user_id, item_ids in self.allPos.items():
            temp[user_id] = item_ids
        self.ui_dict = self.allPos
        self.allPos = temp
        
    def load_text(self, itext_path):
        self.texts = {}
        with open(itext_path, 'r') as fp:
            for line in fp:
                line = line.strip("\n")
                item_id, text = line.split("\t")
                item_id = int(item_id)
                if item_id > self.max_item:
                    continue
                self.texts[item_id] = text
        self.m_items = max(self.texts.keys())+1
        # self.text_array = np.array(temp) # convert to array for easier indexing
                
    def __len__(self):
        return self.num_train
    
    def __get_item__(self, idx):
        pass

    def collate_fcn(self):
        pass
