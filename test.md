# BERT_with_Rec

The basic idea is to use BERT/DistailBERT model as the item encoder in a recommendation scenario. (lookup table for user embedding as usual)

The motivation is to consider recommendation as a down-stream task of transformers, and also add text feature to the Collaborative Filtering.

But because most of the recommendation problems need hundreds of training epochs, so the training/inference is too slow for pratical use. Moreover, such a huge epoch also not fit the recommended setting on the BERT paper. (fine-tune epoch < 5, according to the overfit problem)

    main.py

for train and eval control

    model.py

BERT merged BPR model

    dataset.py

for dataset construction

    sampler.py

sampling utils



