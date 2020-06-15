import jieba
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


raw_data_filename = "./data/jiangye.txt"


for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

(train_data, test_data), info = tfds.load(raw_data_filename, split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                          with_info=True, as_supervised=True)
print(train_data)
