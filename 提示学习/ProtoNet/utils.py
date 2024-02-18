# coding=utf-8

import random
from collections import defaultdict

import tokenization_word as tokenization


class InputExample(object):
    def __init__(self, text_a, text_b=None, label=None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def load_data(file_path):
    class2sents = defaultdict(list)

    with open(file_path, 'r') as in_file:
        lines = in_file.readlines()
        for line in lines:
            sent, class_name = line.strip().split('\t')
            class2sents[class_name].append(sent)
    return class2sents
    
def random_extract_samples(num_ways, num_shots, num_queries, class2sents):
    """ 随机抽取并构建为一个 episode 的数据
    """
    class_names = list(class2sents.keys())
    selected_class_names = random.sample(class_names, num_ways)
    selected_class2sents = dict()

    support_list = []
    query_list = []
    for cls_name in selected_class_names:
        selected_sents = random.sample(class2sents[cls_name], num_shots+num_queries) # 5+5 
        support_list.extend(selected_sents[:num_shots])
        query_list.extend(selected_sents[num_shots:])

    
    return selected_class_names, support_list, query_list

def create_examples(sents):
    examples = []
    for (i, sent) in enumerate(sents):
        text_a = tokenization.convert_to_unicode(sent)
        examples.append(
            InputExample(text_a=text_a))

    return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 0:
            print("*** Example ***")
            print("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids))
    return features