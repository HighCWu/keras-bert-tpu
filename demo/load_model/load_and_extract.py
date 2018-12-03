import os
import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint


if len(sys.argv) != 4:
    print('python load_model.py CONFIG_PATH CHECKPOINT_PATH DICT_PATH')

config_path, checkpoint_path, dict_path = tuple(sys.argv[1:]) # pylint: disable=E0632

model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
model.summary(line_length=120)

if 'COLAB_TPU_ADDR' in os.environ:
    import tensorflow as tf
    tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    strategy = tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
    )
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
    model.compile('adam', 'sparse_categorical_crossentropy')
    batch_size = 8
else:
    batch_size = 1

tokens = ['[CLS]', '语', '言', '模', '型', '[SEP]']

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

token_input = np.asarray([[token_dict[token] for token in tokens] + [0] * (512 - len(tokens)) for i in range(batch_size)])
seg_input = np.asarray([[0] * len(tokens) + [0] * (512 - len(tokens)) for i in range(batch_size)])

print(token_input[0][:len(tokens)])

predicts = model.predict([token_input, seg_input])[0]
for i, token in enumerate(tokens):
    print(token, predicts[i].tolist()[:5])

"""Official outputs:
{
  "linex_index": 0,
  "features": [
    {
      "token": "[CLS]",
      "layers": [
        {
          "index": -1,
          "values": [-0.63251, 0.203023, 0.079366, -0.032843, 0.566809, ...]
        }
      ]
    },
    {
      "token": "语",
      "layers": [
        {
          "index": -1,
          "values": [-0.758835, 0.096518, 1.071875, 0.005038, 0.688799, ...]
        }
      ]
    },
    {
      "token": "言",
      "layers": [
        {
          "index": -1,
          "values": [0.547702, -0.792117, 0.444354, -0.711265, 1.20489, ...]
        }
      ]
    },
    {
      "token": "模",
      "layers": [
        {
          "index": -1,
          "values": [-0.292423, 0.605271, 0.499686, -0.42458, 0.428554, ...]
        }
      ]
    },
    {
      "token": "型",
      "layers": [
        {
          "index": -1,
          "values": [ -0.747346, 0.494315, 0.718516, -0.872353, 0.83496, ...]
        }
      ]
    },
    {
      "token": "[SEP]",
      "layers": [
        {
          "index": -1,
          "values": [-0.874138, -0.216504, 1.338839, -0.105871, 0.39609, ...]
        }
      ]
    }
  ]
}
"""
