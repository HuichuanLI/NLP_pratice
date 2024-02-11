MIN_FLOAT = -3.14e100
MIN_INF = float("-inf")


class Viterbi(object):

    def __init__(self, tags, start_p, trans_p, prev_tags, start_tags, end_tags):
        """
        :param tags: 标签种类
        :param start_p: 初始标签概率
        :param trans_p: 标签转移矩阵
        :param prev_tags: 限制标签转移
        :param start_tags: 限制起始标签
        :param end_tags: 限制终止标签
        """
        # 初始化一些实例变量
        self._tags = tags
        self._start_p = start_p
        self._trans_p = trans_p
        if prev_tags:
            self._prev_tags = prev_tags
        else:
            self._prev_tags = dict([(tag, self._tags) for tag in self._tags])
        if start_tags:
            self._start_tags = start_tags
        else:
            self._start_tags = self._tags
        if end_tags:
            self._end_tags = end_tags
        else:
            self._end_tags = self._tags

    def parse(self, node_scores):
        v = [{}]  # dp矩阵，存放从初始节点到每个节点的最优路径得分
        path = {}  # 记录当前时刻，所有状态选择的路径

        # 解码0时刻
        for t in self._tags:
            t_start_p = 0
            if self._start_p:
                t_start_p = self._start_p.get(t, MIN_FLOAT)
            v[0][t] = node_scores[0][t] + t_start_p  # 0时刻最优路径得分 = 初始标签概率 + 0时刻发射score
            path[t] = [t]  # 0时刻最优路径，即为当前标签

        # 解码1~n时刻，从左往右逐步计算
        for idx in range(1, len(node_scores)):
            v.append({})
            new_path = {}
            for tag in self._tags:  # 迭代所有tag，依次计算初始位置到当前时刻，每个节点的最优路径
                (best_prob, best_tag) = max([  # 动态规划的递推公式
                    (
                        # 上个时刻的最优路径得分 + 当前时刻的转移得分 + 当前时刻的发射得分
                        v[idx - 1][prev_tag] + self._trans_p[prev_tag].get(tag, MIN_FLOAT) + node_scores[idx][tag],
                        prev_tag
                    ) for prev_tag in self._tags if prev_tag[0] in self._prev_tags[tag[0]]  # 迭代每个tag，并确保转移限制
                ])
                v[idx][tag] = best_prob  # 更新最优路径得分
                new_path[tag] = path[best_tag] + [tag]
            path = new_path  # 更新最优路径

        # 解码最终时刻
        (best_prob, best_tag) = max(
            (v[len(node_scores) - 1][tag], tag) for tag in self._tags if tag[0] in self._end_tags  # 迭代每个tag，并确保终止标签限制
        )
        return best_prob, path[best_tag]
