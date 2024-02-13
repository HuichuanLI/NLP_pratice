import copy
import itertools
import re


class EntityScore(object):
    """
    实体级别的precision, recall, f1

    """

    @staticmethod
    def single_entity_score(tags_predict, tags_actual):
        """
        单实体计算score
        :param tags_predict: list of predict tags
        :param tags_actual: list of actual tags
        tags标签体系：BMESO
        """
        assert isinstance(tags_predict[0], list) and isinstance(tags_actual[0], list)  # 确保是内层是个list

        # 2维的拍平到1维，便于计算
        tags_predict = list(itertools.chain(*tags_predict))
        tags_actual = list(itertools.chain(*tags_actual))

        assert len(tags_predict) == len(tags_actual)  # 确保总标签数等长

        # 只截取BIO部分（忽略实体类型部分）
        tags_predict = ''.join(list(map(lambda tag: tag[0], tags_predict)))
        tags_actual = ''.join(list(map(lambda tag: tag[0], tags_actual)))

        # 用正则表达式提取BI*模式，并保存为 startidx_endidx
        regex = re.compile('BI*')
        entity_predict = ['{}_{}'.format(m.start(), m.start() + len(m.group()))
                          for m in regex.finditer(tags_predict)]
        entity_actual = ['{}_{}'.format(m.start(), m.start() + len(m.group()))
                         for m in regex.finditer(tags_actual)]

        # 统计3个数
        predict_num = len(entity_predict)
        actual_num = len(entity_actual)
        correct_num = len(set(entity_predict) & set(entity_actual))

        # 统计准确率、召回率、F1
        precision = float(correct_num) / predict_num if predict_num else 0.
        recall = float(correct_num) / actual_num if actual_num else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.

        return precision, recall, f1

    @staticmethod
    def multi_entities_score(tags_predict, tags_actual, entity_types=('PER', 'LOC', 'ORG', 'GPE')):
        """多实体同时计算score"""
        metric = {}

        for entity_type in entity_types:

            # 将其它实体的标签置为O
            cur_tags_predict, cur_tags_actual = copy.deepcopy(tags_predict), copy.deepcopy(tags_actual)
            for each_pre_tags, each_act_tags in zip(cur_tags_predict, cur_tags_actual):
                assert len(each_pre_tags) == len(each_act_tags)
                for i in range(len(each_pre_tags)):
                    # 预测结果中，其它实体类型的标签置为O
                    if each_pre_tags[i] != 'O' and entity_type not in each_pre_tags[i]:
                        each_pre_tags[i] = 'O'
                    # 实际结果中，其它实体类型的标签置为O
                    if each_act_tags[i] != 'O' and entity_type not in each_act_tags[i]:
                        each_act_tags[i] = 'O'

            # 调用single_entity_score，计算当前实体的score
            p, r, f1 = EntityScore.single_entity_score(cur_tags_predict, cur_tags_actual)
            metric[entity_type] = {'precision': p, 'recall': r, 'f1': f1}

        # 计算宏平均指标
        metric['macro_avg'] = {
            'precision': sum([metric[entity_type]['precision'] for entity_type in entity_types]) / len(entity_types),
            'recall': sum([metric[entity_type]['recall'] for entity_type in entity_types]) / len(entity_types),
        }
        metric['macro_avg']['f1'] = 2 * metric['macro_avg']['precision'] * metric['macro_avg']['recall'] / \
                                    (metric['macro_avg']['precision'] + metric['macro_avg']['recall'])
        return metric
