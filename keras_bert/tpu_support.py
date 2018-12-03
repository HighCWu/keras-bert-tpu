import sys


def tpu_compatible():
    '''Fit the tpu problems we meet while using keras tpu model'''
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.contrib.tpu.python.tpu.keras_support import KerasTPUModel  # pylint: disable=E0611

    def initialize_uninitialized_variables():
        sess = K.get_session()
        uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
        init_op = tf.variables_initializer(
            [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
        )
        sess.run(init_op)

    _tpu_compile = KerasTPUModel.compile

    def tpu_compile(self,
                    optimizer,
                    loss=None,
                    metrics=None,
                    loss_weights=None,
                    sample_weight_mode=None,
                    weighted_metrics=None,
                    target_tensors=None,
                    **kwargs):
        _tpu_compile(self, optimizer, loss, metrics, loss_weights,
                     sample_weight_mode, weighted_metrics,
                     target_tensors, **kwargs)
        initialize_uninitialized_variables()  # for unknown reason, we should run this after compile sometimes
    KerasTPUModel.compile = tpu_compile


def clean_keras_module():
    modules = [i for i in sys.modules.keys()]
    for i in modules:
        if i.split('.')[0] == 'keras':
            del sys.modules[i]


def replace_keras_to_tf_keras():
    clean_keras_module()
    tpu_compatible()
    import tensorflow as tf
    sys.modules['keras'] = tf.keras
    globals()['keras'] = tf.keras
    import keras.backend as K
    K.tf = tf


replace_keras_to_tf_keras()
del sys, tpu_compatible, clean_keras_module, replace_keras_to_tf_keras
