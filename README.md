{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aAnSiPM3rz-G"
      },
      "source": [
        "**Cat and Dog Classification model using Machine learning ALgorithm**\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "executionInfo": {
          "elapsed": 6,
          "status": "ok",
          "timestamp": 1710401092476,
          "user": {
            "displayName": "bc220400651 ZAINEB BIBI",
            "userId": "05963891154143529040"
          },
          "user_tz": -300
        },
        "id": "BWe4cj4IslCH"
      },
      "outputs": [],
      "source": [
        "#Dataset - https://www.kaggle.com/datasets/salader/dogs-vs-cats\n",
        "!mkdir -p ~/.kaggle\n",
        "!cp kaggle.json ~/.kaggle/"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "executionInfo": {
          "elapsed": 30595,
          "status": "ok",
          "timestamp": 1710401128135,
          "user": {
            "displayName": "bc220400651 ZAINEB BIBI",
            "userId": "05963891154143529040"
          },
          "user_tz": -300
        },
        "id": "wPptWUR6sRyo",
        "outputId": "6dee1162-bdf1-4f7c-c0dc-0fb694e51d7b"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /root/.kaggle/kaggle.json'\n",
            "Downloading dogs-vs-cats.zip to /content\n",
            "100% 1.06G/1.06G [00:29<00:00, 39.4MB/s]\n",
            "100% 1.06G/1.06G [00:29<00:00, 38.7MB/s]\n"
          ]
        }
      ],
      "source": [
        "! kaggle datasets download -d salader/dogs-vs-cats"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "executionInfo": {
          "elapsed": 16701,
          "status": "ok",
          "timestamp": 1710401169941,
          "user": {
            "displayName": "bc220400651 ZAINEB BIBI",
            "userId": "05963891154143529040"
          },
          "user_tz": -300
        },
        "id": "LM3WodGYsmOc"
      },
      "outputs": [],
      "source": [
        "import zipfile\n",
        "zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')\n",
        "zip_ref.extractall('/content')\n",
        "zip_ref.close()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "executionInfo": {
          "elapsed": 1378,
          "status": "ok",
          "timestamp": 1710401268677,
          "user": {
            "displayName": "bc220400651 ZAINEB BIBI",
            "userId": "05963891154143529040"
          },
          "user_tz": -300
        },
        "id": "gtJ0BMMXtQJh"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "from keras import Sequential\n",
        "from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "executionInfo": {
          "elapsed": 3377,
          "status": "ok",
          "timestamp": 1710401279403,
          "user": {
            "displayName": "bc220400651 ZAINEB BIBI",
            "userId": "05963891154143529040"
          },
          "user_tz": -300
        },
        "id": "BcBKWDg2t6DB",
        "outputId": "705c03d8-86e1-40a5-a1b8-7c3463d7aa1e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Found 20000 files belonging to 2 classes.\n",
            "Found 5000 files belonging to 2 classes.\n"
          ]
        }
      ],
      "source": [
        "# generators\n",
        "train_ds = keras.utils.image_dataset_from_directory(\n",
        "    directory = '/content/train',\n",
        "    labels='inferred',\n",
        "    label_mode = 'int',\n",
        "    batch_size=32,\n",
        "    image_size=(256,256)\n",
        ")\n",
        "\n",
        "validation_ds = keras.utils.image_dataset_from_directory(\n",
        "    directory = '/content/test',\n",
        "    labels='inferred',\n",
        "    label_mode = 'int',\n",
        "    batch_size=32,\n",
        "    image_size=(256,256)\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 26,
      "metadata": {
        "executionInfo": {
          "elapsed": 4,
          "status": "ok",
          "timestamp": 1710401283353,
          "user": {
            "displayName": "bc220400651 ZAINEB BIBI",
            "userId": "05963891154143529040"
          },
          "user_tz": -300
        },
        "id": "np1wuWJbvT2f"
      },
      "outputs": [],
      "source": [
        "#normalize the numpy data in 0 to 1\n",
        "def process(image,label):\n",
        "    image=tf.cast(image/255.,tf.float32)\n",
        "    return image ,label\n",
        "\n",
        "train_ds=train_ds.map(process)\n",
        "validation_ds=validation_ds.map(process)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "executionInfo": {
          "elapsed": 10,
          "status": "ok",
          "timestamp": 1710402754866,
          "user": {
            "displayName": "bc220400651 ZAINEB BIBI",
            "userId": "05963891154143529040"
          },
          "user_tz": -300
        },
        "id": "RylikLEB3G1j"
      },
      "outputs": [],
      "source": [
        "#Create CNN model and pass the data\n",
        "\n",
        "model=Sequential()\n",
        "model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))\n",
        "model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))\n",
        "\n",
        "model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))\n",
        "model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))\n",
        "\n",
        "model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))\n",
        "model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))\n",
        "\n",
        "model.add(Flatten())\n",
        "model.add(Dense(128,activation='relu'))\n",
        "model.add(Dense(64,activation='relu'))\n",
        "model.add(Dense(1,activation='sigmoid'))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "executionInfo": {
          "elapsed": 32,
          "status": "ok",
          "timestamp": 1710402769493,
          "user": {
            "displayName": "bc220400651 ZAINEB BIBI",
            "userId": "05963891154143529040"
          },
          "user_tz": -300
        },
        "id": "QJxgb4y_8OxZ",
        "outputId": "f8c9dfa9-4287-4af8-d284-995f078df8ac"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Model: \"sequential_1\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " conv2d_1 (Conv2D)           (None, 254, 254, 32)      896       \n",
            "                                                                 \n",
            " max_pooling2d (MaxPooling2  (None, 127, 127, 32)      0         \n",
            " D)                                                              \n",
            "                                                                 \n",
            " conv2d_2 (Conv2D)           (None, 125, 125, 64)      18496     \n",
            "                                                                 \n",
            " max_pooling2d_1 (MaxPoolin  (None, 62, 62, 64)        0         \n",
            " g2D)                                                            \n",
            "                                                                 \n",
            " conv2d_3 (Conv2D)           (None, 60, 60, 128)       73856     \n",
            "                                                                 \n",
            " max_pooling2d_2 (MaxPoolin  (None, 30, 30, 128)       0         \n",
            " g2D)                                                            \n",
            "                                                                 \n",
            " flatten (Flatten)           (None, 115200)            0         \n",
            "                                                                 \n",
            " dense (Dense)               (None, 128)               14745728  \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 64)                8256      \n",
            "                                                                 \n",
            " dense_2 (Dense)             (None, 1)                 65        \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 14847297 (56.64 MB)\n",
            "Trainable params: 14847297 (56.64 MB)\n",
            "Non-trainable params: 0 (0.00 Byte)\n",
            "_________________________________________________________________\n"
          ]
        }
      ],
      "source": [
        "model.summary()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "kTouCOuV80ek"
      },
      "outputs": [],
      "source": [
        "model.compile(optmizier='adam',loss='binary_c')"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "authorship_tag": "ABX9TyNOfUjcYX/X5yOTzVyxoyxK",
      "gpuType": "T4",
      "name": "",
      "version": ""
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
