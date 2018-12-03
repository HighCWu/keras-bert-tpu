import unittest
import os
import tempfile
import random
from tensorflow import keras
import numpy as np
from keras_bert import gelu, get_model, get_custom_objects, get_base_dict, gen_batch_inputs


class TestBERT(unittest.TestCase):

    def test_sample(self):
        model = get_model(
            token_num=200,
            head_num=3,
            transformer_num=2,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'keras_bert_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(
            model_path,
            custom_objects=get_custom_objects(),
        )
        model.summary(line_length=200)

    def test_fit(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_path, 'test_bert_fit.h5')
        sentence_pairs = [
            [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
            [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
            [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
        ]
        token_dict = get_base_dict()
        for pairs in sentence_pairs:
            for token in pairs[0] + pairs[1]:
                if token not in token_dict:
                    token_dict[token] = len(token_dict)
        token_list = list(token_dict.keys())
        if os.path.exists(model_path):
            model = keras.models.load_model(
                model_path,
                custom_objects=get_custom_objects(),
            )
        else:
            model = get_model(
                token_num=len(token_dict),
                head_num=5,
                transformer_num=12,
                embed_dim=25,
                feed_forward_dim=100,
                seq_len=20,
                pos_num=20,
                dropout_rate=0.05,
                attention_activation=gelu,
                lr=1e-3,
            )
        model.summary()

        def _generator():
            while True:
                yield gen_batch_inputs(
                    sentence_pairs,
                    token_dict,
                    token_list,
                    seq_len=20,
                    mask_rate=0.3,
                    swap_sentence_rate=1.0,
                )

        model.fit_generator(
            generator=_generator(),
            steps_per_epoch=1000,
            epochs=1,
            validation_data=_generator(),
            validation_steps=100,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(monitor='val_MLM_loss', factor=0.5, patience=3),
                keras.callbacks.EarlyStopping(monitor='val_MLM_loss', patience=5)
            ],
        )
        # model.save(model_path)
        for inputs, outputs in _generator():
            predicts = model.predict(inputs)
            outputs = list(map(lambda x: np.squeeze(x, axis=-1), outputs))
            predicts = list(map(lambda x: np.argmax(x, axis=-1), predicts))
            batch_size, seq_len = inputs[-1].shape
            for i in range(batch_size):
                for j in range(seq_len):
                    if inputs[-1][i][j]:
                        self.assertEqual(outputs[0][i][j], predicts[0][i][j])
            self.assertTrue(np.allclose(outputs[1], predicts[1]))
            break

    def test_get_layers(self):

        def _custom_layers(x, trainable=True):
            return keras.layers.LSTM(
                units=768,
                trainable=trainable,
                name='LSTM',
            )(x)

        inputs, output_layer = get_model(
            token_num=200,
            embed_dim=768,
            custom_layers=_custom_layers,
            training=False,
        )
        model = keras.models.Model(inputs=inputs, outputs=output_layer)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics={},
        )
        model.summary()
        self.assertTrue(model is not None)
