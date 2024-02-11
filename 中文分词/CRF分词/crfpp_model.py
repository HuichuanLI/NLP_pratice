import re


class CRFPPModel(object):
    """
    加载txt版本的crfpp模型

    可以根据特征模板，获取文本每个位置的特征函数，并计算发射score（累加状态特征函数的权重）
    """
    RE_FEATURE_FUNC = re.compile('%x\\[-?\\d,\\d\\]')  # 解析特征模板的正则（如：U00:%x[-3,0]）
    RE_FEATURE_COLUMN = re.compile('-?\\d')  # 解析特征模板[]内的数字

    def __init__(self, model_path):
        self._model_path = model_path
        self._feature_num = 0

        self._load_text_model()
        self._feature_template_to_func()

    def _load_text_model(self):
        # 初始化模型
        self._model = {
            'tags': [],
            'feature_template': [],
            'feature_func_weight': {},
            'trans_func_weight': {}
        }

        # 一行行读取
        # 状态机，四种状态：head、tags、feature_template、features、weights，分别与txt模型对应
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

                # head状态
                if state == 'head':
                    if line:
                        key, value = line.split(': ')
                        if key == 'maxid':  # 读取maxid，即为总特征数
                            self._feature_num = int(value)
                    else:
                        state = 'tags'  # 遇到空行，此状态结束，进入下一个状态
                        continue

                # tags状态
                if state == 'tags':
                    if line:
                        self._model['tags'].append(line)  # 读取tags(labels)
                    else:
                        state = 'feature_template'
                        tags_len = len(self._model['tags'])
                        trans_weight_id_max = tags_len ** 2 - 1
                        continue

                # feature_template状态
                if state == 'feature_template':
                    if line:
                        self._model['feature_template'].append(line)  # 读取特征模板
                    else:
                        state = 'features'
                        continue

                # features状态
                if state == 'features':
                    if line:
                        offset = line.find(' ')  # 找到空格位置
                        id, feature_func = line[0:offset], line[offset + 1:]  # 划分id与特征函数
                        if id == '0':  # 遇到0，生成转移特征函数（4 * 4）
                            self._model['trans_func_weight'] = {}
                            for tag1 in self._model['tags']:
                                self._model['trans_func_weight'][tag1] = {}
                                for tag2 in self._model['tags']:
                                    self._model['trans_func_weight'][tag1][tag2] = 0
                        else:  # 否则，将状态特征函数添加到feature_funcs
                            feature_funcs.append(feature_func)
                    else:
                        state = 'weights'
                        continue

                # weights状态
                if state == 'weights':
                    weight = float(line)
                    if weight_id <= trans_weight_id_max:  # 转移特征函数权重
                        tag1 = self._model['tags'][weight_id // tags_len]
                        tag2 = self._model['tags'][weight_id % tags_len]
                        self._model['trans_func_weight'][tag1][tag2] = weight
                    else:  # 状态特征函数权重
                        feature_weight_id = weight_id - trans_weight_id_max - 1
                        temp_feature_func_weight.append(weight)
                        if weight_id % tags_len == tags_len - 1:
                            feature_func = feature_funcs[feature_weight_id // tags_len]
                            self._model['feature_func_weight'][feature_func] = tuple(temp_feature_func_weight)
                            temp_feature_func_weight = []
                    weight_id += 1

            del feature_funcs

    def _feature_template_to_func(self):
        """
        将每个特征模板变成一个可供调用的特征函数，供解码（预测）时使用
        （不细讲，感兴趣的同学可以仔细阅读）
        """

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
        for template in self._model['feature_template']:  # 迭代每一个特征模板
            template_new = self.RE_FEATURE_FUNC.sub('{}', template)
            xys = []
            for func in self.RE_FEATURE_FUNC.findall(template):
                x, y = self.RE_FEATURE_COLUMN.findall(func)
                xys.append((int(x), int(y)))
            self._feature_template_func.append(create_feature_func(template_new, xys))

    def gen_feature(self, pairs, pairs_len, idx):
        """
        针对输入文本的特定位置，根据特征模板，生成其对应的所有状态特征函数
        """
        return [
            feature
            # 迭代所有特征模板
            for feature in [template_func(pairs, pairs_len, idx) for template_func in self._feature_template_func]
            # 忽略模型中不存在的特征函数
            if feature in self._model['feature_func_weight']
        ]

    def get_tags(self):
        return self._model['tags']

    def get_trans_func_weight(self):
        return self._model['trans_func_weight']

    def compute_score(self, features):
        """
        根据生成的特征函数，从模型中取相应的权重，并累加，构成发射score
        """
        score = {}
        for idx, tag in enumerate(self._model['tags']):
            score[tag] = sum([self._model['feature_func_weight'][feature][idx] for feature in features])
        return score


if __name__ == '__main__':
    model = CRFPPModel('model/crf-seg.model.txt')
    print(model.get_tags())

    # 计算文本中每个位置的发射score（累加状态特征函数的权重）
    content = '今天天气不错哈'
    for i in range(len(content)):
        print('\ncur index:', i)
        features = model.gen_feature(content, len(content), i)  # 针对位置i，根据模板生成其状态特征函数
        print('features:\n', '  '.join(features))
        score = model.compute_score(features)
        print('score: ', score)
