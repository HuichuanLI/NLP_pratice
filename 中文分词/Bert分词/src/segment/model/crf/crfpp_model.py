import json
import re
from datetime import datetime

from segment import logger


class CRFPPModel(object):
    RE_FEATURE_FUNC = re.compile('%x\\[-?\\d,\\d\\]')
    RE_FEATURE_COLUMN = re.compile('-?\\d')

    def __init__(self, model_path, is_json=False):
        self._model_path = model_path
        self._feature_num = 0
        start_time = datetime.now()
        logger.debug('begin loading CRF model')
        if is_json:
            self._load_json_model()
        else:
            self._load_text_model()
        self._feature_template_to_func()
        logger.debug('end loading CRF model, cost time: {}'.format(datetime.now() - start_time))

    def _load_text_model(self):
        self._model = {'tags': [], 'feature_template': [], 'feature_func_weight': {}, 'trans_func_weight': {}}
        state = 'head'
        with open(self._model_path, encoding="utf-8") as model_file:
            feature_funcs = []
            weight_id = 0
            trans_weight_id_max = 0
            line_num = 0
            tags_len = 0
            temp_feature_func_weight = []
            for line in model_file:
                line_num += 1
                line = line.strip()
                if state == 'head':
                    if line:
                        key, value = line.split(': ')
                        if key == 'maxid':
                            self._feature_num = int(value)
                    else:
                        state = 'tags'
                        continue
                if state == 'tags':
                    if line:
                        self._model['tags'].append(line)
                    else:
                        state = 'feature_template'
                        tags_len = len(self._model['tags'])
                        trans_weight_id_max = tags_len ** 2 - 1
                        continue
                if state == 'feature_template':
                    if line:
                        self._model['feature_template'].append(line)
                    else:
                        state = 'features'
                        continue
                if state == 'features':
                    if line:
                        offset = line.find(' ')
                        id, feature_func = line[0:offset], line[offset + 1:]
                        if id == '0':
                            self._model['trans_func_weight'] = {}
                            for tag1 in self._model['tags']:
                                self._model['trans_func_weight'][tag1] = {}
                                for tag2 in self._model['tags']:
                                    self._model['trans_func_weight'][tag1][tag2] = 0
                        else:
                            feature_funcs.append(feature_func)
                    else:
                        state = 'weights'
                        continue
                if state == 'weights':
                    weight = float(line)
                    if weight_id <= trans_weight_id_max:
                        tag1 = self._model['tags'][weight_id // tags_len]
                        tag2 = self._model['tags'][weight_id % tags_len]
                        self._model['trans_func_weight'][tag1][tag2] = weight
                    else:
                        feature_weight_id = weight_id - trans_weight_id_max - 1
                        temp_feature_func_weight.append(weight)
                        if weight_id % tags_len == tags_len - 1:
                            feature_func = feature_funcs[feature_weight_id // tags_len]
                            self._model['feature_func_weight'][feature_func] = tuple(temp_feature_func_weight)
                            temp_feature_func_weight = []
                    weight_id += 1
            del feature_funcs

    def _load_json_model(self):
        with open(self._model_path, encoding='utf-8') as model_file:
            self._model = json.load(model_file)

    def _feature_template_to_func(self):

        def get_feature(pairs, pairs_len, idx, x, y):
            feature_idx = idx + x
            if feature_idx < 0:
                return '_B-{}'.format(-feature_idx)
            elif feature_idx >= pairs_len:
                return '_B+{}'.format(feature_idx - pairs_len + 1)
            else:
                return pairs[feature_idx][y]

        def create_feature_func(template_str, xys):

            def feature_func(pairs, pairs_len, idx):
                return template_str.format(*[get_feature(pairs, pairs_len, idx, x, y) for x, y in xys])

            return feature_func

        self._feature_template_func = []
        for template in self._model['feature_template']:
            template_new = self.RE_FEATURE_FUNC.sub('{}', template)
            xys = []
            for func in self.RE_FEATURE_FUNC.findall(template):
                x, y = self.RE_FEATURE_COLUMN.findall(func)
                xys.append((int(x), int(y)))
            self._feature_template_func.append(create_feature_func(template_new, xys))

    def gen_feature(self, pairs, pairs_len, idx):
        return [
            feature
            for feature in [template_func(pairs, pairs_len, idx) for template_func in self._feature_template_func]
            if feature in self._model['feature_func_weight']
        ]

    def get_tags(self):
        return self._model['tags']

    def get_trans_func_weight(self):
        return self._model['trans_func_weight']

    def compute_score(self, features):
        score = {}
        for idx, tag in enumerate(self._model['tags']):
            score[tag] = sum([self._model['feature_func_weight'][feature][idx] for feature in features])
        return score

    def dump_bin_model(self, output_bin_model_path):
        with open(output_bin_model_path, 'w', encoding='utf-8') as model_file:
            json.dump(self._model, model_file, ensure_ascii=False)
