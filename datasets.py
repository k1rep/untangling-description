import json
import torch
from torch.utils.data import Dataset
from typing import Dict
import transformers
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class SupervisedDataset(Dataset):
    """A dataset for supervised fine-tuning."""
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_length: int):
        super(SupervisedDataset, self).__init__()

        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[idx],
            labels=self.labels[idx],
            attention_mask=self.attention_mask[idx],
        )


class LazySupervisedDataset(Dataset):
    """A dataset for supervised fine-tuning (Lazy)."""
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_length: int):
        super(LazySupervisedDataset, self).__init__()

        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if idx in self.cached_data_dict:
            return self.cached_data_dict[idx]
        ret = preprocess([self.raw_data[idx]["conversations"]], self.tokenizer, self.max_length)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[idx] = ret
        return ret


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, max_length: int,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_length=max_length)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_length=max_length)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int,
        system_message: str = "system",
) -> Dict:

    # im_start = tokenizer.im_start_id
    # im_end = tokenizer.im_end_id

    begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
    start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
    end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
    eot_id = tokenizer.get_vocab()["<|eot_id|>"]
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids
    _user = tokenizer('user').input_ids
    _assistant = tokenizer('assistant').input_ids

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        input_id, target = [], []
        system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(system_message).input_ids + [eot_id]
        input_id += system
        target += [IGNORE_TOKEN_ID] * len(input_id)
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = sentence["from"]
            value = sentence["value"]
            if role == 'user':
                _input_id = [start_header_id] + _user + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [
                    eot_id]
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
            elif role == 'assistant':
                _input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [
                    eot_id]
                _target = [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(_assistant) + \
                          [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(nl_tokens) + tokenizer(value).input_ids + [eot_id]
            else:
                raise NotImplementedError
            input_id += _input_id
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_length - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_length - len(target))
        input_ids.append(input_id[:max_length])
        targets.append(target[:max_length])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
