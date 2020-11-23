import tensorflow as tf
import matplotlib.pyplot as plt

from data import x, y


def main():
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(32, drop_remainder=True)

    model_dir = r"models\model_10"
    try:
        model = tf.keras.models.load_model(model_dir)
    except OSError:
        model = create_and_train_model(dataset, model_dir)

    x_eval = tf.range(-15.0, 15.0, 0.1)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_eval)
    test_dataset = test_dataset.batch(32)
    y_predict = model.predict(test_dataset)

    plt.figure()
    plt.plot(x, y, ".")
    plt.plot(x_eval.numpy(), y_predict, "-")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.show()


def create_and_train_model(dataset, model_dir):
    depth = 6
    width = 64

    model = tf.keras.models.Sequential([tf.keras.layers.Dense(width, activation="relu", input_shape=(1, ))])
    for _ in range(depth - 1):
        model.add(tf.keras.layers.Dense(width, activation="relu"))
    model.add(tf.keras.layers.Dense(1))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanAbsoluteError(),
        )

    model.fit(
        dataset,
        epochs=512,
        callbacks=[tf.keras.callbacks.EarlyStopping("loss", patience=32, restore_best_weights=True)],
        )

    model.save(model_dir)

    return model


if __name__ == '__main__':
    main()
