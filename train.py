import tensorflow as tf
import matplotlib.pyplot as plt

from data import x, y


def main():
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(32, drop_remainder=True)

    model_dir = r"models\model_03"
    try:
        model = tf.keras.models.load_model(model_dir)
    except OSError:
        model = create_and_train_model(dataset, model_dir)

    x_eval = tf.range(-10.0, 10.0, 0.1)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_eval)
    test_dataset = test_dataset.batch(32)
    y_predict = model.predict(test_dataset)

    plt.figure()
    plt.plot(x, y, ".")
    plt.plot(x_eval.numpy(), y_predict, "-")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


def create_and_train_model(dataset, model_dir):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation="relu", input_shape=(1,)),
        tf.keras.layers.Dense(1),
        ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError(),
        )

    model.fit(
        dataset,
        epochs=100,
        callbacks=[tf.keras.callbacks.EarlyStopping("loss", patience=5)],
        )

    model.save(model_dir)

    return model


if __name__ == '__main__':
    main()
