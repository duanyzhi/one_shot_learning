import tensorflow as tf

# 计算cosine distance
def cd(memory_images, x_target):
    cosine_distance = []
    """
    memory_images: ways*shot个[batch_size, 64]
    support_image：[32, 64]  sum_support:[32, 1]
    support_magnitude：[32，1]
    dot_product:    [32, 1, 1]
    cosine_similarity: [32, 1]
    """
    for support_image in tf.unstack(memory_images, axis=0):
        sum_support = tf.reduce_sum(tf.square(support_image), 1, keep_dims=True)  # 先平方，再按行求和（一个batch求和）
        # rsqrt： Computes reciprocal of square root of x element-wise
        support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support, 1e-10, float("inf")))  # 将值变到固定大小
        # print("1",support_magnitude.shape)
        #   expand_dims： add a batch dimension to a single element
        k1 = tf.expand_dims(x_target, 1)  # [32,1,64]
        k2 = tf.expand_dims(support_image, 2)  # [32,64,1]
        # print("kk",k1.shape, k2.shape)
        dot_product = tf.matmul(k1, k2)
        dot_product = tf.squeeze(dot_product, [1, ])  # [32, 1]
        cosine_similarity = dot_product * support_magnitude  #
        # print(2, cosine_similarity.shape)
        cosine_distance.append(cosine_similarity)
    similarities = tf.concat(axis=1, values=cosine_distance)           # size: [batch_size, cfg.ways*cfg.shot]
    softmax_a = tf.nn.softmax(similarities)
    return softmax_a, similarities

def compute_distance(memory_images, x_target):
    cosine_distance = []
    r = 0
    """
    support_image：[32, 64]  sum_support:[32, 1]
    support_magnitude：[32，1]
    dot_product:    [32, 1, 1]
    cosine_similarity: [32, 1]
    """
    for support_image in tf.unstack(memory_images, axis=0):
        sum_support = tf.reduce_sum(tf.square(support_image), 1, keep_dims=True)  # 先平方，再按行求和（一个batch求和）
        sum_target = tf.reduce_sum(tf.square(support_image), 1, keep_dims=True)  # 先平方，再按行求和（一个batch求和）
        # rsqrt： Computes reciprocal of square root of x element-wise
        support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support, 1e-10, float("inf")))  # 将值变到固定大小
        target_magnitude = tf.rsqrt(tf.clip_by_value(sum_target, 1e-10, float("inf")))  # 将值变到固定大小
        # print("1",support_magnitude.shape)
        #   expand_dims： add a batch dimension to a single element
        k1 = tf.expand_dims(x_target, 1)                                # [32,1,64]
        k2 = tf.expand_dims(support_image, 2)                           # [32,64,1]
        # print("kk",k1.shape, k2.shape)
        dot_product = tf.matmul(k1, k2)
        dot_product = tf.squeeze(dot_product, [1, ])                    # [32, 1]
        cosine_similarity = dot_product / (support_magnitude*target_magnitude)  #
        cosine_distance.append(cosine_similarity)
        # print(2, cosine_similarity.shape)                               # [32, 1]
        a = tf.nn.softmax(cosine_similarity)
        r += a*support_image                                            # [32, 64]
        # print(r.shape)
    similarities = tf.concat(axis=1, values=cosine_distance)           # [32,5]
    softmax_a = tf.nn.softmax(similarities)
    return softmax_a, similarities, r

# ------------------------------------------------------------------------------
