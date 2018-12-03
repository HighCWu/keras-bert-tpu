from keras.layers import Dense


class CompatibilityDense(Dense):
    def get_config(self):
        base_config = super(CompatibilityDense, self).get_config()
        base_config['activation'] = self.activation
        return base_config
