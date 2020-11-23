import tensorflow as tf
from tensorflow.keras import layers, losses
import matplotlib.pyplot as plt

from data import x, y


def main():
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(32, drop_remainder=True)

    model_dir = r"models\model_01"
    try:
        model = tf.keras.models.load_model(model_dir)
    except OSError:
        model = create_and_train_model(dataset, model_dir)

    x_test = tf.range(-10.0, 10.0, 0.1)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.batch(32)
    y_predict = model.predict(test_dataset)

    plt.figure()
    plt.plot(x, y, ".")
    plt.plot(x_test.numpy(), y_predict, "-")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


def create_and_train_model(dataset, model_dir):
    model = tf.keras.models.Sequential([
        layers.Dense(1, input_shape=(1, )),
        ])

    model.compile(
        optimizer="adam",
        loss=losses.MeanSquaredError(),
        )

    model.fit(dataset, epochs=10)

    model.save(model_dir)

    return model


if __name__ == '__main__':
    main()
