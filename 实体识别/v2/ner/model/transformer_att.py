"""
tf版的多头注意力实现

仅用来理解下Transformer的实现，不实际生产使用下面代码
"""
import tensorflow as tf


def scaled_dot_product_attention(q, k, v):
    """
    缩放点积注意力
    """
    # Q K点积
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # dk即为K的最后一个维度
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # attention_weights和V相乘，产生输出
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


# 我们定义一个MultiHeadAttention层
class MultiHeadAttention(tf.keras.layers.Layer):
    """
    多头注意力
    """
    def __init__(self, d_model, num_heads):
        """
        d_model: embedding维度，论文为512
        num_heads: 头的个数，论文为8
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0  # 确保能整除

        self.depth = d_model // self.num_heads  # depth即为dk，由d_model整除num_heads得到

        # 分别定义Q K V的投影层
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        # 定义最后的dense层
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        划分多头

        分拆最后一个维度d_model到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # 对Q K V进行投影
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # 对Q K V划分多头
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # 并行计算多个Q K V的缩放点积注意力
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        # 通过reshape，将attention的结果拼接起来
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # 将拼接后的结果输入全连接层，产生输出
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
