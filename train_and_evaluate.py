import numpy as np
import tensorflow as tf
from munch import Munch
import shutil
import os
from tensorflow.python.keras.engine import data_adapter


class MyModel(tf.keras.Model):

    def __init__(self, params, *args, **kwargs):
        self.params = params
        super().__init__(*args, **kwargs)

    def get_global_step(self, training):
        return self.optimizer.iterations

    def custom_ops(self, features, labels, sample_weight, training):

        # use the below line and return if overriding predict_step
        # predictions = self.call(features, training=training)

        params = self.params
        summary_writer = params.summary_writer_train if training else params.summary_writer_eval
        global_step = self.get_global_step(training)
        tf.summary.experimental.set_step(global_step)

        with summary_writer.as_default():
            tf.summary.scalar("test_scalar", tf.reduce_sum(features[0]))
            tf.summary.scalar("iterations", global_step)

        # Calculate loss watching model.trainable_variables
        # tape.watch(model.trainable_variables)
        with tf.GradientTape() as tape:
            predictions = self(features, training=training)
            with summary_writer.as_default():
                # custom loss_fn
                # loss = loss_fn(predictions, labels, params)
                loss = self.compiled_loss(predictions, labels, sample_weight, self.losses)
                if self.losses:  # add regularization losses
                    loss = loss + tf.add_n(self.losses)

        # apply gradients and do backprop
        if training:
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        summary_writer.flush()
        self.compiled_metrics.update_state(predictions, labels, sample_weight)
        r_dict = {m.name: m.result() for m in self.metrics}
        r_dict["loss"] = loss
        return r_dict

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        features, labels, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        return self.custom_ops(features, labels, sample_weight, training=True)

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        features, labels, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        return self.custom_ops(features, labels, sample_weight, training=False)


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
    model = MyModel(params=params, inputs=input_layer, outputs=dense_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.evaluate(inp_valid, out)
    model.fit(inp, out, epochs=10)
    model.evaluate(inp_valid, out)
    model.fit(inp, out, epochs=10)
    model.evaluate(inp_valid, out)


if __name__ == '__main__':
    main()
