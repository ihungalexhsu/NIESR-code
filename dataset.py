import torch 
from torch.utils.data import Dataset
import os
import pickle 
import numpy as np

class PickleDataset(Dataset):
    def __init__(self, pickle_path, config=None, sort=True):
        with open(pickle_path, 'rb') as f:
            self.data_dict = pickle.load(f)

        # remove the utterance out of limit
        self.keys = self.get_keys(config, sort=sort)

    def get_keys(self, config, sort):
        if config:
            max_feature_length = config['max_feature_length']
            min_feature_length = config['min_feature_length']
            max_text_length = config['max_text_length']
            min_text_length = config['min_text_length']
            keys = [key for key in self.data_dict 
                    if self.data_dict[key]['feature'].shape[0] <= max_feature_length and 
                    self.data_dict[key]['feature'].shape[0] >= min_feature_length and 
                    len(self.data_dict[key]['token_ids']) <= max_text_length and 
                    len(self.data_dict[key]['token_ids']) >= min_text_length]
        else:
            keys = [key for key in self.data_dict]

        # sort by feature length
        if sort:
            keys = sorted(keys, key=lambda x: self.data_dict[x]['feature'].shape[0])
        return keys

    def __getitem__(self, index):
        utt_id = self.keys[index]
        feature = self.data_dict[utt_id]['feature'].astype(np.float32)
        token_ids = self.data_dict[utt_id]['token_ids']
        speaker_ids = self.data_dict[utt_id]['speaker_ids']
        env_ids = self.data_dict[utt_id]['env_ids']
        transcript = self.data_dict[utt_id]['Transcript']
        return feature, token_ids, speaker_ids, env_ids, transcript

    def __len__(self):
        return len(self.keys)

