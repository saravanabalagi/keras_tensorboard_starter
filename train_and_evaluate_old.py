import numpy as np
import tensorflow as tf
from munch import Munch
import shutil
import os


class MyModel(tf.keras.Model):

    def __init__(self, params, *args, **kwargs):
        self.params = params
        self.params.summary_writer = params.summary_writer_eval
        super().__init__(*args, **kwargs)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=False, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
            use_multiprocessing=False, **kwargs):
        self.params.summary_writer = self.params.summary_writer_train
        return_val = super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data,
                                 shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps,
                                 validation_freq, max_queue_size, workers, use_multiprocessing, **kwargs)
        self.params.summary_writer = self.params.summary_writer_eval
        return return_val

    def call(self, inputs, training=False):
        with self.params.summary_writer.as_default():
            tf.summary.scalar("test_scalar", tf.reduce_sum(inputs[0]), step=self.optimizer.iterations)
            tf.summary.scalar("iterations", self.optimizer.iterations, step=self.optimizer.iterations)
        print(self._layers)
        next_layer_inputs = inputs
        for layer in self._layers:
            print(layer.__dict__)
            output = layer(next_layer_inputs)
            next_layer_inputs = output
        return next_layer_inputs


def main():
    params = Munch(model_dir='experiments/')

    # Make an empty dir, clear contents for testing
    os.makedirs(params.model_dir, exist_ok=True)
    for directory in os.listdir(params.model_dir):
        shutil.rmtree(os.path.join(params.model_dir, directory))

    params.summary_writer_train = tf.summary.create_file_writer(f"{params.model_dir}/train")
    params.summary_writer_eval = tf.summary.create_file_writer(f"{params.model_dir}/eval")

    inp = np.random.randint(0, 10, (5, 3)).astype(np.float32)
    inp_valid = np.random.randint(1000, 1100, (5, 3)).astype(np.float32)
    out = np.ones((5, 4), dtype=np.float32)
    print(inp, inp_valid)
    print(np.sum(inp, axis=1), np.sum(inp_valid, axis=1))

    input_layer = tf.keras.layers.Input(shape=inp.shape)
    dense_layer = tf.keras.layers.Dense(4, name='Dense')(input_layer)
    model = MyModel(inputs=input_layer, outputs=dense_layer, params=params)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.evaluate(inp_valid, out)
    model.fit(inp, out, epochs=10)
    model.evaluate(inp_valid, out)
    model.fit(inp, out, epochs=10)
    model.evaluate(inp_valid, out)


if __name__ == '__main__':
    main()
