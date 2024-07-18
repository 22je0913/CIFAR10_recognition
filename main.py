import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets, layers, models, losses


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


file = r"C:\Users\Kush\Downloads\data\cifar-10-batches-py\data_batch_1"
data_batch_1 = unpickle(file)
train_images = data_batch_1[b"data"]
train_images = train_images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
train_labels = data_batch_1[b"labels"]
train_labels = np.array(train_labels)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))
model.compile(
    optimizer="adam",
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
history = model.fit(
    train_images, train_labels, epochs=10, validation_data=(train_images, train_labels)
)
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.show()
test_loss, test_acc = model.evaluate(train_images, train_labels, verbose=2)
