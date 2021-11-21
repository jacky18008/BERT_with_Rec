import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer, DistilBertTokenizerFast, DistilBertPreTrainedModel

class IEBPR(DistilBertPreTrainedModel):
    def __init__(self, 
                 config,
                 model_config,
                 dataset, 
                 model_path=None):
        super(IEBPR, self).__init__(config)
        self.encoder = model_config["encoder"]
        self.latent_dim = model_config["latent_dim"]
        self.dataset = dataset
        self.use_device = model_config["device"]
        self.dropout = torch.nn.Dropout(p=0.1)
        self.text_maxlen = model_config["text_maxlen"]
        self.batch_size = model_config["batch_size"]
        if model_path is None:
            self.model_path = "distilbert-base-uncased"
        self.distilbert = DistilBertModel(config) # = DistilBertModel.from_pretrained(model_path) # use distil bert as encoder
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_path)
        self.linear = nn.Linear(config.hidden_size, self.latent_dim, bias=False)
        self.init_weights()
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.embedding_user = torch.nn.Embedding( # lookup table for user embedding
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
            )
        # set non-trainable item lookup table for convenience
        with torch.no_grad():
            self.embedding_table_item = torch.nn.Embedding(
                num_embeddings=self.num_items, embedding_dim=self.latent_dim
            )
            self.embedding_table_item.weight.requires_grad=False
        # use encoder to dynamiclly predict on item embedding
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        print('use NORMAL distribution initilizer')
    
    def set_item_embed_table(self):
        self.embedding_table_item.weight.requires_grad=False

    def embedding_item(self, item):
        # predict on BERT for a batch of items
        item_text = [self.dataset.texts[i] for i in item]
        item_meta = self.tokenizer(item_text, return_tensors="pt", padding=True, \
                                    truncation=True) # input_id, attention_mask, ...
        item_embed = self.item_encode(**item_meta, batch_item_id=item)
        return item_embed

    def bpr_loss(self, users_emb, pos_emb, neg_emb):
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users_emb))

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss
    
    def forward(self, users, pos, neg):
        users_emb = self.embedding_user(users)
        pos_emb = self.embedding_item(pos)
        neg_emb = self.embedding_item(neg)
        return users_emb, pos_emb, neg_emb

    def item_encode(self, input_ids, attention_mask, batch_item_id, **kwargs):
        # print("input_ids", input_ids, "attention_mask", attention_mask)
        input_ids, attention_mask = input_ids.to(self.use_device), attention_mask.to(self.use_device)
        item_embed = self.distilbert(input_ids, attention_mask=attention_mask)[0][:, 0] # [CLS] embedding
        item_embed = self.linear(item_embed)
        item_embed = torch.nn.functional.normalize(item_embed, p=2, dim=-1) # control the vector length
        # record item text embedding
        self.embedding_table_item.weight[batch_item_id] = item_embed
        return  item_embed 

    def item_chunk_encode(self, input_ids, attention_mask): # have to standarize the num of chunk/padding
        item_embeds = []
        for input_ids_, attention_mask_ in zip(input_ids, attention_mask):
            item_chunk_embed = self.distilbert(input_ids_, attention_mask_)[0]
            item_chunk_embed = self.linear(item_chunk_embed)
            item_embeds.append(item_chunk_embed)
        item_embeds = torch.mean(torch.stack(item_embeds, dim=0), dim=0)
        return item_embeds
    
    def getUsersRating(self, users):
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_table_item.weight
        rating = nn.functional.sigmoid(torch.matmul(users_emb, items_emb.t()))
        return rating
    


    
    