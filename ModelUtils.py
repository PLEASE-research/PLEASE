import os
from sklearn.utils import compute_class_weight
import numpy as np
class ModelUtils:
    @staticmethod
    def create_directories(directories):

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    @staticmethod
    def get_loss_weight(labels, weight_dict):

        label_list = labels.numpy().squeeze().tolist()
        weight_list = []

        for lab in label_list:
            if lab == 0:
                weight_list.append(weight_dict['clean'])
            else:
                weight_list.append(weight_dict['defect'])

        weight_tensor = torch.tensor(weight_list).reshape(-1, 1)
        return weight_tensor

    @staticmethod
    def compute_class_weights(train_label):

        sample_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_label),
            y=train_label
        )

        weight_dict = {
            'defect': np.max(sample_weights),
            'clean': np.min(sample_weights)
        }
        return weight_dict

    @staticmethod
    def collate_fn(batch):

        file_data = [data_list for data_list in batch]
        return file_data
