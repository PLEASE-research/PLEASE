import torch
import torch.nn as nn
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
from PLEASE import *
from my_util import *
from PredictionUtils import *




class InputFeatures:


    def __init__(self, input_ids, label):
        self.input_ids = input_ids
        self.label = label


class TextDataset(Dataset):


    def __init__(self, tokenizer, block_size, datasets, labels):
        self.examples = []
        labels = torch.FloatTensor(labels)
        for dataset, label in zip(datasets, labels):
            dataset_ids = [PredictionUtils.convert_examples_to_features(item, tokenizer, block_size)
                           for item in dataset]
            self.examples.append(InputFeatures(dataset_ids, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), self.examples[i].label


class PredictionConfig:


    def __init__(self):

        self.save_model_dir = 'output/model/PLEASE/'
        self.prediction_dir = 'output/prediction/PLEASE/'


        self.embed_dim = 768
        self.hidden_dim = 128
        self.layers = 2
        self.output_dim = 256
        self.dropout = 0.2

        self.model_type = 'roberta'
        self.model_name_or_path = './datasets/Bert_base'
        self.config_name = None
        self.tokenizer_name = './datasets/Bert_base'
        self.cache_dir = None
        self.block_size = 75
        self.do_lower_case = False


        self.seed = 0
        self.drop_length = 35000
        self.limit_length = 1000

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DefectPredictor:


    def __init__(self, config: PredictionConfig):
        self.config = config
        self.utils = PredictionUtils()
        self._setup_environment()
        self._load_bert_components()

    def _setup_environment(self):

        self.utils.set_seed(self.config.seed)
        self.utils.create_directories([self.config.prediction_dir])

    def _load_bert_components(self):

        MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.config.model_type]

        # 加载配置
        self.bert_config = config_class.from_pretrained(
            self.config.config_name if self.config.config_name else self.config.model_name_or_path,
            cache_dir=self.config.cache_dir if self.config.cache_dir else None
        )

        # 加载分词器
        self.tokenizer = tokenizer_class.from_pretrained(
            self.config.tokenizer_name,
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

    def _load_trained_model(self, model_path):

        model = PLEASE(
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.layers,
            output_dim=self.config.output_dim,
            dropout=self.config.dropout,
            device=self.config.device
        )

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.config.device)
        model.eval()

        return model

    def _process_file_data(self, df):

        df = df[df['code_line'].ne('')]

        file_label = bool(df['file-label'].unique())
        line_label = df['line-label'].tolist()
        line_number = df['line_number'].tolist()
        is_comments = df['is_comment'].tolist()
        code = df['code_line'].tolist()

        return {
            'file_label': file_label,
            'line_label': line_label,
            'line_number': line_number,
            'is_comments': is_comments,
            'code': code
        }

    def _predict_file(self, model, file_data):

        code = file_data['code']
        file_label = file_data['file_label']


        code2d = prepare_code2d(code, True)
        code3d = [code2d]
        codevec = TextDataset(self.tokenizer, self.config.block_size, code3d, [file_label])

        sig = nn.Sigmoid()

        with torch.no_grad():
            input_tensor = torch.tensor(codevec.examples[0].input_ids)


            input_parts = self.utils.split_input_for_memory(input_tensor, self.config.limit_length)


            cov_input = []
            for item in input_parts:
                bert_output = self.codebert(
                    item.to(self.config.device),
                    attention_mask=item.to(self.config.device).ne(1)
                ).pooler_output
                cov_input.append(bert_output)

            cov_input = torch.cat(cov_input, dim=0)

            output, line_att_weight = model([cov_input])

            file_prob = sig(output).item()
            prediction = bool(round(file_prob))

        torch.cuda.empty_cache()

        return file_prob, prediction, line_att_weight

    def _create_result_rows(self, dataset_name, train_rel, test_rel, filename,
                            file_data, file_prob, prediction, line_att_weight):

        row_list = []
        numpy_line_attn = line_att_weight[0].cpu().detach().numpy()

        code = file_data['code']
        line_label = file_data['line_label']
        line_number = file_data['line_number']
        is_comments = file_data['is_comments']
        file_label = file_data['file_label']

        for i in range(len(code)):
            row_dict = {
                'project': dataset_name,
                'train': train_rel,
                'test': test_rel,
                'filename': filename,
                'file-level-ground-truth': file_label,
                'prediction-prob': file_prob,
                'prediction-label': prediction,
                'line-number': line_number[i],
                'line-level-ground-truth': line_label[i],
                'is-comment-line': is_comments[i],
                'code-line': code[i],
                'line-attention-score': numpy_line_attn[i]
            }
            row_list.append(row_dict)

        return row_list

    def predict_release(self, dataset_name, test_release):

        print(f'Generate a release version {test_release} pass {dataset_name} Prediction')

        model_path = self.config.save_model_dir + dataset_name + '/best_model.pth'
        model = self._load_trained_model(model_path)

        train_rel = all_train_releases[dataset_name]
        test_df = get_df(test_release)

        row_list = []


        for filename, df in tqdm(test_df.groupby('filename')):

            file_data = self._process_file_data(df)


            if self.utils.should_skip_file(file_data['code'], self.config.drop_length):
                continue

            # 预测文件缺陷
            file_prob, prediction, line_att_weight = self._predict_file(model, file_data)


            file_rows = self._create_result_rows(
                dataset_name, train_rel, test_release, filename,
                file_data, file_prob, prediction, line_att_weight
            )
            row_list.extend(file_rows)

        # 保存结果
        result_df = pd.DataFrame(row_list)
        output_path = self.config.prediction_dir + test_release + '.csv'
        result_df.to_csv(output_path, index=False)
        print(f'Complete the release version {test_release}')

        return result_df

    def predict_dataset(self, dataset_name):
        print(f"Start predicting the dataset: {dataset_name}")

        test_releases = all_eval_releases[dataset_name][1:]  # 跳过验证集，只处理测试集

        results = {}
        for test_rel in test_releases:
            results[test_rel] = self.predict_release(dataset_name, test_rel)

        return results

    def predict_all_datasets(self):

        dataset_names = list(all_releases.keys())
        all_results = {}

        for dataset_name in dataset_names:
            all_results[dataset_name] = self.predict_dataset(dataset_name)

        return all_results


class PredictionManager:

    def __init__(self, config: PredictionConfig = None):
        if config is None:
            config = PredictionConfig()
        self.predictor = DefectPredictor(config)

    def predict_single_dataset(self, dataset_name):

        return self.predictor.predict_dataset(dataset_name)

    def predict_single_release(self, dataset_name, release_name):

        return self.predictor.predict_release(dataset_name, release_name)

    def predict_all(self):

        return self.predictor.predict_all_datasets()



if __name__ == "__main__":

    config = PredictionConfig()
    config.prediction_dir = 'output/prediction/'

    manager = PredictionManager(config)

    results = manager.predict_single_dataset("activemq")
