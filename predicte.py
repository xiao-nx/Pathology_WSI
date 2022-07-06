def predict(self, text_a, text_b):
    """

    :param text_a:
    :param text_b:
    :return:
    """

    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    input_ids, input_mask, segment_ids = self.convert_single_example(text_a, text_b)

    features = collections.OrderedDict()
    features['input_ids'] = create_int_feature(input_ids)
    features['input_mask'] = create_int_feature(input_mask)
    features['segment_ids'] = create_int_feature(segment_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features)) # 将feature转换为example

    self.writer.write(tf_example.SerializeToString())# 序列化example，写入tfrecord文件

    result = self.estimator.predict(input_fn=self.predict_input_fn)