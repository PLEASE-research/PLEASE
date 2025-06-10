import copy
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.utils import compute_class_weight
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from PLEASE import *
from my_util import *
from itertools import dropwhile
from ModelUtils import  *


class InputFeatures:


    def __init__(self, input_ids, label):
        self.input_ids = input_ids
        self.label = label


class TextDataset(Dataset):


    def __init__(self, tokenizer, args, datasets, labels):
        self.examples = []
        labels = torch.FloatTensor(labels)
        for dataset, label in zip(datasets, labels):
            dataset_ids = [self._convert_examples_to_features(item, tokenizer, args) for item in dataset]
            self.examples.append(InputFeatures(dataset_ids, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), self.examples[i].label

    def _convert_examples_to_features(self, item, tokenizer, args):

        code = ' '.join(item)
        code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        return source_ids


class ModelConfig:
    """模型配置类"""

    def __init__(self):
        # 数据相关配置
        self.file_lvl_gt = 'datasets/preprocessed_data/'
        self.save_model_dir = 'output/model/PLEASE/'
        self.loss_dir = 'output/loss/PLEASE/'

        # 训练相关配置
        self.batch_size = 16
        self.num_epochs = 10
        self.lr = 0.0001
        self.max_grad_norm = 1.0
        self.max_train_LOC = 900
        self.dropout = 0.2
        self.weight_decay = 0.0

        # 模型架构配置
        self.embed_dim = 768
        self.hidden_dim = 128
        self.layers = 2
        self.output_dim = 256
        self.use_layer_norm = True

        # BERT相关配置
        self.model_type = 'roberta'
        self.model_name_or_path = './datasets/Bert_base'
        self.config_name = None
        self.tokenizer_name = './datasets/Bert_base'
        self.cache_dir = None
        self.block_size = 75
        self.do_lower_case = False

        # 设备配置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DefectPredictionTrainer:


    def __init__(self, config: ModelConfig):
        self.config = config
        self.utils = ModelUtils()
        self._setup_environment()
        self._load_bert_components()

    def _setup_environment(self):

        self.utils.set_seed(self.config.seed)

    def _load_bert_components(self):

        MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.config.model_type]


        self.bert_config = config_class.from_pretrained(
            self.config.config_name if self.config.config_name else self.config.model_name_or_path,
            cache_dir=self.config.cache_dir if self.config.cache_dir else None
        )


        self.tokenizer = tokenizer_class.from_pretrained(
            self.config.tokenizer_name,
            from_tf=True,
            do_lower_case=self.config.do_lower_case,
            cache_dir=self.config.cache_dir if self.config.cache_dir else None
        )


        if self.config.block_size <= 0:
            self.config.block_size = self.tokenizer.max_len_single_sentence
        self.config.block_size = min(self.config.block_size, self.tokenizer.max_len_single_sentence)


        if self.config.model_name_or_path:
            self.codebert = model_class.from_pretrained(
                self.config.model_name_or_path,
                from_tf=bool('.ckpt' in self.config.model_name_or_path),
                config=self.bert_config,
                cache_dir=self.config.cache_dir if self.config.cache_dir else None
            )
        else:
            self.codebert = model_class(self.bert_config)

        self.codebert.to(self.config.device)

    def _prepare_data(self, dataset_name):

        train_rel = all_train_releases[dataset_name]
        valid_rel = all_eval_releases[dataset_name][0]

        train_df = get_df(train_rel)
        valid_df = get_df(valid_rel)

        train_code3d, train_label = get_code3d_and_label(train_df, True, self.config.max_train_LOC)
        valid_code3d, valid_label = get_code3d_and_label(valid_df, True, self.config.max_train_LOC)

        return train_code3d, train_label, valid_code3d, valid_label

    def _create_data_loaders(self, train_code3d, train_label, valid_code3d, valid_label):

        x_train_vec = TextDataset(self.tokenizer, self.config, train_code3d, train_label)
        x_valid_vec = TextDataset(self.tokenizer, self.config, valid_code3d, valid_label)

        train_dl = DataLoader(
            x_train_vec,
            shuffle=True,
            batch_size=self.config.batch_size,
            drop_last=True,
            collate_fn=self.utils.collate_fn
        )
        valid_dl = DataLoader(
            x_valid_vec,
            shuffle=False,
            batch_size=self.config.batch_size,
            drop_last=False,
            collate_fn=self.utils.collate_fn
        )

        return train_dl, valid_dl

    def _initialize_model_and_optimizer(self):

        model = PLEASE(
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.layers,
            output_dim=self.config.output_dim,
            dropout=self.config.dropout,
            device=self.config.device
        )
        model.to(self.config.device)

        optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.lr
        )

        criterion = nn.BCEWithLogitsLoss()
        sig = nn.Sigmoid()

        return model, optimizer, criterion, sig

    def _train_epoch(self, model, train_dl, optimizer, criterion, weight_dict):

        model.train()
        train_losses = []

        for step, batch in tqdm(enumerate(train_dl), total=len(train_dl), desc='Train Loop'):
            inputs = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            labels = torch.tensor(labels)

            # 获取CodeBERT编码
            cov_inputs = []
            with torch.no_grad():
                for item in inputs:
                    cov_inputs.append(
                        self.codebert(item.to(self.config.device),
                                      attention_mask=item.to(self.config.device).ne(1)).pooler_output
                    )

            # 设置损失权重
            weight_tensor = self.utils.get_loss_weight(labels, weight_dict)
            criterion.weight = weight_tensor.to(self.config.device)

            # 前向传播
            output, _ = model(cov_inputs)
            loss = criterion(output, labels.reshape(self.config.batch_size, 1).to(self.config.device))
            train_losses.append(loss.item())

            # 反向传播
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()

        return np.mean(train_losses)

    def _validate_epoch(self, model, valid_dl, criterion, sig):
        model.eval()
        val_losses = []
        outputs = []
        outputs_labels = []

        with torch.no_grad():
            criterion.weight = None

            for step, batch in tqdm(enumerate(valid_dl), total=len(valid_dl), desc='Valid Loop'):
                inputs = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                labels = torch.tensor(labels)


                cov_inputs = []
                for item in inputs:
                    cov_inputs.append(
                        self.codebert(item.to(self.config.device),
                                      attention_mask=item.to(self.config.device).ne(1)).pooler_output
                    )


                output, _ = model(cov_inputs)
                outputs.append(sig(output))
                outputs_labels.append(labels)

                val_loss = criterion(output, labels.reshape(len(labels), 1).to(self.config.device))
                val_losses.append(val_loss.item())

        y_prob = torch.cat(outputs)
        y_gt = torch.cat(outputs_labels)
        valid_auc = roc_auc_score(y_gt, y_prob.to('cpu'))

        return np.mean(val_losses), valid_auc

    def _save_model_and_losses(self, dataset_name, best_model, best_epoch, optimizer,
                               train_loss_all_epochs, val_loss_all_epochs, val_auc_all_epochs):

        actual_save_model_dir = self.config.save_model_dir + dataset_name + '/'

        # 保存最佳模型
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, actual_save_model_dir + 'best_model.pth')

        # 保存损失记录
        loss_df = pd.DataFrame({
            'epoch': np.arange(1, len(train_loss_all_epochs) + 1),
            'train_loss': train_loss_all_epochs,
            'valid_loss': val_loss_all_epochs,
            'valid_auc': val_auc_all_epochs
        })
        loss_df.to_csv(self.config.loss_dir + dataset_name + '-loss_record.csv', index=False)

    def train_model(self, dataset_name):

        print(f"Start training the dataset: {dataset_name}")

        # 创建必要目录
        actual_save_model_dir = self.config.save_model_dir + dataset_name + '/'
        self.utils.create_directories([actual_save_model_dir, self.config.loss_dir])

        # 准备数据
        train_code3d, train_label, valid_code3d, valid_label = self._prepare_data(dataset_name)

        # 计算类别权重
        weight_dict = self.utils.compute_class_weights(train_label)

        # 创建数据加载器
        train_dl, valid_dl = self._create_data_loaders(train_code3d, train_label, valid_code3d, valid_label)

        # 初始化模型和优化器
        model, optimizer, criterion, sig = self._initialize_model_and_optimizer()

        # 训练循环
        best_auc = 0
        best_epoch = 0
        best_model = None
        train_loss_all_epochs = []
        val_loss_all_epochs = []
        val_auc_all_epochs = []

        model.zero_grad()

        for epoch in range(1, self.config.num_epochs + 1):
            # 训练
            train_loss = self._train_epoch(model, train_dl, optimizer, criterion, weight_dict)
            train_loss_all_epochs.append(train_loss)

            # 验证
            val_loss, valid_auc = self._validate_epoch(model, valid_dl, criterion, sig)
            val_loss_all_epochs.append(val_loss)
            val_auc_all_epochs.append(valid_auc)

            # 更新最佳模型
            if valid_auc >= best_auc:
                best_model = copy.deepcopy(model)
                best_auc = valid_auc
                best_epoch = epoch

            print(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, AUC: {valid_auc:.4f}')

            # 保存模型和损失记录
            if epoch % self.config.num_epochs == 0:
                print(f'{dataset_name} Training Completed!')
                self._save_model_and_losses(
                    dataset_name, best_model, best_epoch, optimizer,
                    train_loss_all_epochs, val_loss_all_epochs, val_auc_all_epochs
                )

    def train_all_datasets(self, start_dataset="activemq"):

        dataset_names = list(all_releases.keys())
        for dataset_name in dropwhile(lambda x: x != start_dataset, dataset_names):
            self.train_model(dataset_name)


# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = ModelConfig()

    # 创建训练器
    trainer = DefectPredictionTrainer(config)

    # 开始训练
    trainer.train_all_datasets(start_dataset="activemq")