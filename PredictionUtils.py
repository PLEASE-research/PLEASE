class PredictionUtils:
    @staticmethod
    def create_directories(directories):

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    @staticmethod
    def convert_examples_to_features(item, tokenizer, block_size):
        code = ' '.join(item)
        code_tokens = tokenizer.tokenize(code)[:block_size - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        return source_ids

    @staticmethod
    def should_skip_file(code, drop_length=35000):

        return len(code) > drop_length

    @staticmethod
    def split_input_for_memory(input_tensor, limit_length=1000):

        return input_tensor.split(limit_length, 0)
