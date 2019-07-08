import numpy as np
from torch.utils.data import DataLoader
import torch
from dataset import PickleDataset

def _collate_fn(l):
    l.sort(key=lambda x: x[0].shape[0], reverse=True)
    features = [torch.from_numpy(feature).float() for feature, _,_,_,_ in l]
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    ilens = [feature.shape[0] for feature, _,_,_,_ in l]
    texts = [torch.from_numpy(np.array(text)) for _, text,_,_,_ in l]
    spks = []
    envs = []
    trans = []
    for _, _, spk, env, tran in l:
        spks.append(spk)
        envs.append(env)
        trans.append(tran)

    return padded_features, ilens, texts, spks, envs, trans

def get_data_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn, num_workers=0)
