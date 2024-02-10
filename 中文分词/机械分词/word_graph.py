class Node(object):
    """
    定义词图中单个节点
    """

    def __init__(self, key, weight, source=None, freq=None):
        self.key = key  # key, 即词
        self.weight = weight  # 节点权重
        self.source = source  # 节点来源，比如core_dict、user_dict 或 model_dict等（后面会讲）
        self.freq = freq  # 节点词频

    def __str__(self):
        return ''.join(['Node(key=', self.key, ',weight=', str(self.weight), ',source=', str(self.source),
                        ',freq=', str(self.freq), ')'])


class WordGraph(object):
    """
    定义词图结构

    使用两个数组来表示图：节点列表 + 终止节点列表
    """

    NODE_S = Node('#S#', 0)  # start节点，权重为0
    NODE_E = Node('#E#', 0)  # end节点，权重为0

    def __init__(self):
        self._start_node_list = []  # 所有可能的节点（词）列表
        self._end_nodes_index_list = []  # 对应的终止节点index，表示有边相连
        self._size = 0

    def __len__(self):
        return self._size

    def insert_start_word(self, node):
        """
        插入节点
        """
        self._start_node_list.append(node)
        self._size += 1

    def insert_end_words(self, row):
        """
        插入终止节点，即创建边
        """
        self._end_nodes_index_list.append(row)

    def __str__(self):
        graph_list = []
        for i in range(self._size):
            graph_list.extend([str(self._start_node_list[i]), '  =>  '])
            graph_list.extend([str(node) for node in list(map(self.get_node, self._end_nodes_index_list[i]))])
            graph_list.append('\n')
        return ''.join(graph_list)

    def get_node(self, index):
        """
        根据索引获取对应node
        """
        return self._start_node_list[index]

    def calculate(self):
        """
        计算最优路径

        动态规划，从后往前，逐步计算 终止节点到每个节点的最优路径（权重最高路径）
        dp[i] = max(weight + dp[i+1][1], weight + dp[i+1][2], ..., weight + dp[i+1][j], ...)
        每个节点的最优路径，等于所有 其节点权重+终止节点最优路径 中的最大的
        """
        self._start_node_list.append(WordGraph.NODE_E)

        # 终止节点到自己，最长路径为0，选择节点为自己
        route = {self._size: (0, self._size)}  # (最优路径, 选择的终止节点)

        for i in range(self._size - 1, -1, -1):
            route[i] = max(
                (self._start_node_list[i].weight + route[index][0], index) for index in self._end_nodes_index_list[i]
            )

        return route


if __name__ == '__main__':
    graph = WordGraph()

    graph.insert_start_word(WordGraph.NODE_S)  # 0
    graph.insert_start_word(Node('我', 1, 'core_dict'))  # 1
    graph.insert_start_word(Node('喜', 2, 'core_dict'))  # 2
    graph.insert_start_word(Node('喜欢', 4, 'user_dict'))  # 3
    graph.insert_start_word(Node('欢', 1, 'core_dict'))  # 4

    graph.insert_end_words([1])
    graph.insert_end_words([2, 3])
    graph.insert_end_words([4])
    graph.insert_end_words([5])
    graph.insert_end_words([5])

    route = graph.calculate()

    print('词图：\n', graph)
    print('最优路径：\n', route)
