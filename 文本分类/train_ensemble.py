import time

import jieba
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier


def read_data(data_path):
    """
    读取原始数据，分词，返回titles、labels
    """
    titles, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        print('current file:', data_path)
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            _, _, label, title, _ = line.split('_!_')
            words = ' '.join(jieba.cut(title))  # 分词
            titles.append(words), labels.append(label)
        print(data_path, 'finish')
    return titles, labels


def get_features(feature_num=10000):
    """分词，使用词袋法、获得tf-idf特征"""
    # 读取train dev test数据
    train_path, dev_path, test_path = \
        'data/toutiao_cat_data.train.txt', 'data/toutiao_cat_data.dev.txt', 'data/toutiao_cat_data.test.txt'
    (train_titles, train_labels), (dev_titles, dev_labels), (test_titles, test_labels) = \
        read_data(train_path), read_data(dev_path), read_data(test_path)

    # 设置TfidfVectorizer，并训练
    vectorizer = TfidfVectorizer(max_features=feature_num)  # 设定max_features，保留词频最高的feature_num个词
    vectorizer.fit(train_titles)  # 输入训练集分词结果，通过fit方法拟合vectorizer

    # 打印前后100个特征词
    print('head 100 words:', ' '.join(vectorizer.get_feature_names()[:100]))
    print('tail 100 words:', ' '.join(vectorizer.get_feature_names()[-100:]))

    # 转换titles为tfidf矩阵
    # 通过transform方法将分好词的text转化为tfidf的float矩阵
    train_X, dev_X, test_X = \
        vectorizer.transform(train_titles), vectorizer.transform(dev_titles), vectorizer.transform(test_titles)
    print('shape:', train_X.shape, dev_X.shape, test_X.shape)

    return train_X, train_labels, dev_X, dev_labels, test_X, test_labels


def train_random_forest():
    """训练随机森林分类器"""
    train_X, train_labels, dev_X, dev_labels, test_X, test_labels = get_features(feature_num=10000)

    # 为RF设置参数，并训练
    clf = RandomForestClassifier(
        n_estimators=100,  # 设置森林里决策树的个数为100个
        criterion='gini',  # 设置使用gini指数决定划分属性
        max_depth=200,  # 树的最大深度，None为不限制树深，控制模型拟合程度
        max_features='sqrt',  # 每次选择最优时，考虑的最大特征数，即sqrt(n_features)
        bootstrap=True,  # 使用自助采样，获得样本子集
        class_weight='balanced',  # 平衡根据类别频率，平衡权重
        random_state=0,  # 设置随机种子，可复现的随机
        n_jobs=-1,  # 启用所有cpu，并行训练
    )
    init_time = time.time()
    clf.fit(train_X, train_labels)
    print('train rf finish, cost time: {}s'.format(time.time() - init_time))

    # 评估train acc、dev acc、test acc
    train_acc = clf.score(train_X, train_labels)
    dev_acc = clf.score(dev_X, dev_labels)
    test_acc = clf.score(test_X, test_labels)
    print('train acc:', train_acc)
    print('dev acc:', dev_acc)
    print('test acc:', test_acc)


def train_gbdt():
    """训练gbdt模型"""
    train_X, train_labels, dev_X, dev_labels, test_X, test_labels = get_features(feature_num=10000)

    # 为GBDT设置参数，并训练
    clf = GradientBoostingClassifier(
        n_estimators=100,  # 设置森林里决策树的个数为100个
        learning_rate=0.1,  # 学习率
        loss='deviance',  # 损失函数，deviance即偏差、残差
        subsample=1.0,  # 训练个体学习器时，可以允许采样的百分比（类似bagging算法，带来样本的扰动），默认为1.0，表示不采样
        max_depth=100,  # 设置最大树深
    )
    init_time = time.time()
    clf.fit(train_X, train_labels)
    print('train gbdt finish, cost time: {}s'.format(time.time() - init_time))

    # 评估train acc、dev acc、test acc
    train_acc = clf.score(train_X, train_labels)
    dev_acc = clf.score(dev_X, dev_labels)
    test_acc = clf.score(test_X, test_labels)
    print('train acc:', train_acc)
    print('dev acc:', dev_acc)
    print('test acc:', test_acc)


def train_xgboost():
    """训练xgb"""
    train_X, train_labels, dev_X, dev_labels, test_X, test_labels = get_features(feature_num=10000)

    clf = XGBClassifier(
        n_estimators=100,  # 设置森林决策树为100棵
        learning_rate=0.1,  # 学习率
        booster='gbtree',  # 个体学习器类型为gbtree，即CART决策树
        objective='multi:softmax',  # 目标，多分类softmax
        max_depth=100,  # 设置最大树深为100
        subsample=1.0,  # 训练个体学习器时，可以允许采样的百分比（类似bagging算法，带来样本的扰动），默认为1.0，表示不采样
        reg_lambda=1,  # l2正则化系数，与正则化强度成正比（与lr svm的C互为倒数）
        random_state=0,  # 固定随机种子
        n_jobs=-1,  # 启用所有cpu，并行训练
    )
    init_time = time.time()
    clf.fit(train_X, train_labels)
    print('train xgboost finish, cost time: {}s'.format(time.time() - init_time))

    # 评估train acc、dev acc、test acc
    train_acc = clf.score(train_X, train_labels)
    dev_acc = clf.score(dev_X, dev_labels)
    test_acc = clf.score(test_X, test_labels)
    print('train acc:', train_acc)
    print('dev acc:', dev_acc)
    print('test acc:', test_acc)


if __name__ == '__main__':
    train_random_forest()
    # train_gbdt()
    train_xgboost()
