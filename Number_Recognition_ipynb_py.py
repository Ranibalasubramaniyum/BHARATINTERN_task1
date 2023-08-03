{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LYJJsOAyLKe1"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yDF4ACpBLKe2",
        "outputId": "7674c26c-a550-4d94-fcd6-6e6657bd8b3b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n",
            "11490434/11490434 [==============================] - 0s 0us/step\n"
          ]
        }
      ],
      "source": [
        "# Load MNIST dataset\n",
        "mnist = tf.keras.datasets.mnist\n",
        "(train_images, train_labels), (test_images, test_labels) = mnist.load_data()\n",
        "\n",
        "# Normalize the pixel values to range [0, 1]\n",
        "train_images, test_images = train_images / 255.0, test_images / 255.0\n",
        "\n",
        "# Reshape the images to have a single channel (grayscale) and convert labels to categorical\n",
        "train_images = train_images.reshape(-1, 28, 28, 1)\n",
        "test_images = test_images.reshape(-1, 28, 28, 1)\n",
        "\n",
        "# One-hot encode the labels\n",
        "train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)\n",
        "test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kD5GE7egLKe3",
        "outputId": "ae8f6600-c3d2-4af0-933c-17eca11ecc5e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/5\n",
            "1500/1500 [==============================] - 72s 44ms/step - loss: 0.1591 - accuracy: 0.9513 - val_loss: 0.0542 - val_accuracy: 0.9837\n",
            "Epoch 2/5\n",
            "1500/1500 [==============================] - 60s 40ms/step - loss: 0.0503 - accuracy: 0.9842 - val_loss: 0.0474 - val_accuracy: 0.9844\n",
            "Epoch 3/5\n",
            "1500/1500 [==============================] - 59s 40ms/step - loss: 0.0361 - accuracy: 0.9889 - val_loss: 0.0392 - val_accuracy: 0.9874\n",
            "Epoch 4/5\n",
            "1500/1500 [==============================] - 58s 39ms/step - loss: 0.0272 - accuracy: 0.9912 - val_loss: 0.0449 - val_accuracy: 0.9890\n",
            "Epoch 5/5\n",
            "1500/1500 [==============================] - 58s 39ms/step - loss: 0.0200 - accuracy: 0.9933 - val_loss: 0.0466 - val_accuracy: 0.9863\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7de89a80bca0>"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ],
      "source": [
        "# Build the model\n",
        "model = tf.keras.Sequential([\n",
        "    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),\n",
        "    tf.keras.layers.MaxPooling2D((2, 2)),\n",
        "    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),\n",
        "    tf.keras.layers.MaxPooling2D((2, 2)),\n",
        "    tf.keras.layers.Flatten(),\n",
        "    tf.keras.layers.Dense(64, activation='relu'),\n",
        "    tf.keras.layers.Dense(10, activation='softmax')\n",
        "])\n",
        "\n",
        "# Compile the model\n",
        "model.compile(optimizer='adam',\n",
        "              loss='categorical_crossentropy',\n",
        "              metrics=['accuracy'])\n",
        "\n",
        "# Train the model\n",
        "model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_split=0.2)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jf9PNUowLKe4",
        "outputId": "d9da27fe-1051-40bc-99af-5df329819606"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Test accuracy: 98.84%\n"
          ]
        }
      ],
      "source": [
        "# Evaluate the model on the test dataset\n",
        "test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)\n",
        "print(f'Test accuracy: {test_accuracy * 100:.2f}%')\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 470
        },
        "id": "WEYxzxWpLKe4",
        "outputId": "4512ad0f-2797-43f3-de00-676a91c615be"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 122ms/step\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAaAAAAGzCAYAAABpdMNsAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAmIElEQVR4nO3deXBW9b3H8c9DSB4ihCeE7BCyIFtlsbJEhCBIhkUvyuJF0RmhekFssAUqVnrLZq/mXhyro8NSp0r0NkClZSm0ZaqRhFo2QRG5lRRiFLghbJY8IUjA5Hf/yPBcH5IAJzwPvyS8XzNnhuec8z3nm5NDPjnn+eU8LmOMEQAAN1gL2w0AAG5OBBAAwAoCCABgBQEEALCCAAIAWEEAAQCsIIAAAFYQQAAAKwggAIAVBBCaDZfLpYULF9puo1EZOnSohg4d6nv95ZdfyuVyKScnx1pPl7u8R9w8CCDUaenSpXK5XEpPT2/wNkpKSrRw4ULt3bs3cI0FyZQpU+Ryueqd/vd//9fxNvPz8/22ERoaqrS0ND322GP64osvgvBVBM+2bdu0cOFCnTlzxnYrfnJycq74fcvNzbXdIq6gpe0G0Djl5uYqJSVFu3bt0qFDh3Trrbc63kZJSYkWLVqklJQU3X777YFvMoCefPJJZWZm+s0zxmj69OlKSUlRhw4dGrztH/3oR+rfv78uXryojz/+WG+88Yb++Mc/6rPPPlNiYuL1tu5IcnKyvvnmG4WGhjqq27ZtmxYtWqQpU6YoMjIyOM01wJAhQ/Tf//3ftea/8sor+vTTTzV8+HALXeFaEUCopbi4WNu2bdPatWv15JNPKjc3VwsWLLDdVlANHDhQAwcO9Jv34Ycf6ty5c3r00Ueva9sZGRl68MEHJUk/+MEP1LVrV/3oRz/S22+/rblz59ZZU1FRodatW1/XfuvicrnUqlWrgG/XlrS0NKWlpfnN++abb/TDH/5Q99xzj+Lj4y11hmvBLTjUkpubq3bt2um+++7Tgw8+WO9tjDNnzmjWrFlKSUmR2+1Wx44d9dhjj+nUqVPKz89X//79JdX80L10S+TSew8pKSmaMmVKrW1e/n7AhQsXNH/+fPXt21cej0etW7dWRkaGtmzZck1fy4EDB3T48GFHX/8lK1eulMvl0iOPPNKg+vrcc889kmqCXpIWLlwol8ulv//973rkkUfUrl07DR482Lf+b37zG/Xt21fh4eGKiorSww8/rCNHjtTa7htvvKHOnTsrPDxcAwYM0F//+tda69T3HtCBAwc0ceJExcTEKDw8XN26ddO///u/+/qbM2eOJCk1NdX3vfzyyy+D0qMkHT58WAcOHLjCUazfxo0bVV5eft2/OCD4uAJCLbm5uRo/frzCwsI0adIkLVu2TB999JEvUCTp7NmzysjI0Oeff67HH39cd9xxh06dOqU//OEPOnr0qHr06KHnn39e8+fP17Rp05SRkSFJuuuuuxz14vV69etf/1qTJk3S1KlTVV5erjfffFMjR47Url27rnprr0ePHrr77ruVn5/vaL8XL17Uu+++q7vuukspKSmOaq+mqKhIktS+fXu/+f/6r/+qLl266MUXX9SlT0l54YUXNG/ePE2cOFH/9m//ppMnT+r111/XkCFD9Mknn/huh7355pt68sknddddd2nmzJn64osvdP/99ysqKkpJSUlX7Gffvn3KyMhQaGiopk2bppSUFBUVFWnjxo164YUXNH78eP3jH//QqlWr9Morryg6OlqSFBMTE7QeH3vsMRUUFKghnxaTm5ur8PBwjR8/3nEtbjADfMfu3buNJPPee+8ZY4yprq42HTt2ND/+8Y/91ps/f76RZNauXVtrG9XV1cYYYz766CMjyaxYsaLWOsnJyWby5Mm15t99993m7rvv9r3+9ttvTWVlpd86//znP01cXJx5/PHH/eZLMgsWLKg177vbu1YbN240kszSpUsd116yZcsWI8m89dZb5uTJk6akpMT88Y9/NCkpKcblcpmPPvrIGGPMggULjCQzadIkv/ovv/zShISEmBdeeMFv/meffWZatmzpm3/hwgUTGxtrbr/9dr9j9cYbb9T6+ouLi2t9T4YMGWIiIiLMV1995befS99HY4x56aWXjCRTXFwc9B6NqTkPGvLj6fTp0yYsLMxMnDjRcS1uPG7BwU9ubq7i4uI0bNgwSTXvGTz00ENavXq1qqqqfOv9/ve/V58+fTRu3Lha23C5XAHrJyQkRGFhYZKk6upqff311/r222/Vr18/ffzxx1etN8Y4vvqRam6/hYaGauLEiY5rL/f4448rJiZGiYmJuu+++1RRUaG3335b/fr181tv+vTpfq/Xrl2r6upqTZw4UadOnfJN8fHx6tKli+825O7du3XixAlN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AIAVBBAAwAoCCABgBQEEALCipe0GEFhTp051XNO7d+8G7evzzz93XNOjRw/HNXfccYfjmqFDhzqukaQ777zTcc2RI0cc1yQlJTmuuZG+/fZbxzUnT550XJOQkOC4piEOHz7coDoeRhpcXAEBAKwggAAAVhBAAAArCCAAgBUEEADACgIIAGAFAQQAsIIAAgBYQQABAKwggAAAVhBAAAArCCAAgBU8jLSZycvLuyE1DbV58+Ybsp927do1qO722293XLNnzx7HNf3793dccyOdP3/ecc0//vEPxzUNeaBtVFSU45qioiLHNQg+roAAAFYQQAAAKxwH0NatWzVmzBglJibK5XJp/fr1fsuNMZo/f74SEhIUHh6uzMxMHTx4MFD9AgCaCccBVFFRoT59+mjJkiV1Ll+8eLFee+01LV++XDt37lTr1q01cuTIBt1TBgA0X44HIYwePVqjR4+uc5kxRq+++qp+/vOf64EHHpAkvfPOO4qLi9P69ev18MMPX1+3AIBmI6DvARUXF6u0tFSZmZm+eR6PR+np6dq+fXudNZWVlfJ6vX4TAKD5C2gAlZaWSpLi4uL85sfFxfmWXS47O1sej8c3JSUlBbIlAEAjZX0U3Ny5c1VWVuabjhw5YrslAMANENAAio+PlyQdP37cb/7x48d9yy7ndrvVtm1bvwkA0PwFNIBSU1MVHx/v95f1Xq9XO3fu1MCBAwO5KwBAE+d4FNzZs2d16NAh3+vi4mLt3btXUVFR6tSpk2bOnKn/+I//UJcuXZSamqp58+YpMTFRY8eODWTfAIAmznEA7d69W8OGDfO9nj17tiRp8uTJysnJ0bPPPquKigpNmzZNZ86c0eDBg7V582a1atUqcF0DAJo8lzHG2G7iu7xerzwej+02ADg0YcIExzXvvvuu45r9+/c7rvnuL81OfP311w2qQ42ysrIrvq9vfRQcAODmRAABAKwggAAAVhBAAAArCCAAgBUEEADACgIIAGAFAQQAsIIAAgBYQQABAKwggAAAVhBAAAArCCAAgBWOP44BQPMXGxvruGbp0qWOa1q0cP478PPPP++4hqdaN05cAQEArCCAAABWEEAAACsIIACAFQQQAMAKAggAYAUBBACwggACAFhBAAEArCCAAABWEEAAACsIIACAFTyMFEAtWVlZjmtiYmIc1/zzn/90XFNYWOi4Bo0TV0AAACsIIACAFQQQAMAKAggAYAUBBACwggACAFhBAAEArCCAAABWEEAAACsIIACAFQQQAMAKAggAYAUPIwWasUGDBjWo7rnnngtwJ3UbO3as45r9+/cHvhFYwRUQAMAKAggAYAUBBACwggACAFhBAAEArCCAAABWEEAAACsIIACAFQQQAMAKAggAYAUBBACwggACAFjBw0iBZuzee+9tUF1oaKjjmry8PMc127dvd1yD5oMrIACAFQQQAMAKxwG0detWjRkzRomJiXK5XFq/fr3f8ilTpsjlcvlNo0aNClS/AIBmwnEAVVRUqE+fPlqyZEm964waNUrHjh3zTatWrbquJgEAzY/jQQijR4/W6NGjr7iO2+1WfHx8g5sCADR/QXkPKD8/X7GxserWrZueeuopnT59ut51Kysr5fV6/SYAQPMX8AAaNWqU3nnnHeXl5em//uu/VFBQoNGjR6uqqqrO9bOzs+XxeHxTUlJSoFsCADRCAf87oIcfftj37169eql3797q3Lmz8vPzNXz48Frrz507V7Nnz/a99nq9hBAA3ASCPgw7LS1N0dHROnToUJ3L3W632rZt6zcBAJq/oAfQ0aNHdfr0aSUkJAR7VwCAJsTxLbizZ8/6Xc0UFxdr7969ioqKUlRUlBYtWqQJEyYoPj5eRUVFevbZZ3Xrrbdq5MiRAW0cANC0OQ6g3bt3a9iwYb7Xl96/mTx5spYtW6Z9+/bp7bff1pkzZ5SYmKgRI0boF7/4hdxud+C6BgA0eS5jjLHdxHd5vV55PB7bbQCNTnh4uOOaDz/8sEH7uu222xzX3HPPPY5rtm3b5rgGTUdZWdkV39fnWXAAACsIIACAFQQQAMAKAggAYAUBBACwggACAFhBAAEArCCAAABWEEAAACsIIACAFQQQAMAKAggAYAUBBACwIuAfyQ0gOObMmeO45vvf/36D9rV582bHNTzZGk5xBQQAsIIAAgBYQQABAKwggAAAVhBAAAArCCAAgBUEEADACgIIAGAFAQQAsIIAAgBYQQABAKwggAAAVvAwUsCC++67z3HNvHnzHNd4vV7HNZL0/PPPN6gOcIIrIACAFQQQAMAKAggAYAUBBACwggACAFhBAAEArCCAAABWEEAAACsIIACAFQQQAMAKAggAYAUBBACwgoeRAtepffv2jmtee+01xzUhISGOa/70pz85rpGkHTt2NKgOcIIrIACAFQQQAMAKAggAYAUBBACwggACAFhBAAEArCCAAABWEEAAACsIIACAFQQQAMAKAggAYAUBBACwgoeRAt/RkAd+bt682XFNamqq45qioiLHNfPmzXNcA9woXAEBAKwggAAAVjgKoOzsbPXv318RERGKjY3V2LFjVVhY6LfO+fPnlZWVpfbt26tNmzaaMGGCjh8/HtCmAQBNn6MAKigoUFZWlnbs2KH33ntPFy9e1IgRI1RRUeFbZ9asWdq4caPWrFmjgoIClZSUaPz48QFvHADQtDkahHD5m605OTmKjY3Vnj17NGTIEJWVlenNN9/UypUrdc8990iSVqxYoR49emjHjh268847A9c5AKBJu673gMrKyiRJUVFRkqQ9e/bo4sWLyszM9K3TvXt3derUSdu3b69zG5WVlfJ6vX4TAKD5a3AAVVdXa+bMmRo0aJB69uwpSSotLVVYWJgiIyP91o2Li1NpaWmd28nOzpbH4/FNSUlJDW0JANCENDiAsrKytH//fq1evfq6Gpg7d67Kysp805EjR65rewCApqFBf4g6Y8YMbdq0SVu3blXHjh198+Pj43XhwgWdOXPG7yro+PHjio+Pr3Nbbrdbbre7IW0AAJowR1dAxhjNmDFD69at0wcffFDrr7n79u2r0NBQ5eXl+eYVFhbq8OHDGjhwYGA6BgA0C46ugLKysrRy5Upt2LBBERERvvd1PB6PwsPD5fF49MQTT2j27NmKiopS27Zt9fTTT2vgwIGMgAMA+HEUQMuWLZMkDR061G/+ihUrNGXKFEnSK6+8ohYtWmjChAmqrKzUyJEjtXTp0oA0CwBoPlzGGGO7ie/yer3yeDy228BNqmvXro5rDhw4EIROanvggQcc12zcuDEInQDXpqysTG3btq13Oc+CAwBYQQABAKwggAAAVhBAAAArCCAAgBUEEADACgIIAGAFAQQAsIIAAgBYQQABAKwggAAAVhBAAAArCCAAgBUN+kRUoLFLTk5uUN1f/vKXAHdStzlz5jiu2bRpUxA6AezhCggAYAUBBACwggACAFhBAAEArCCAAABWEEAAACsIIACAFQQQAMAKAggAYAUBBACwggACAFhBAAEArOBhpGiWpk2b1qC6Tp06BbiTuhUUFDiuMcYEoRPAHq6AAABWEEAAACsIIACAFQQQAMAKAggAYAUBBACwggACAFhBAAEArCCAAABWEEAAACsIIACAFQQQAMAKHkaKRm/w4MGOa55++ukgdAIgkLgCAgBYQQABAKwggAAAVhBAAAArCCAAgBUEEADACgIIAGAFAQQAsIIAAgBYQQABAKwggAAAVhBAAAAreBgpGr2MjAzHNW3atAlCJ3UrKipyXHP27NkgdAI0LVwBAQCsIIAAAFY4CqDs7Gz1799fERERio2N1dixY1VYWOi3ztChQ+Vyufym6dOnB7RpAEDT5yiACgoKlJWVpR07dui9997TxYsXNWLECFVUVPitN3XqVB07dsw3LV68OKBNAwCaPkeDEDZv3uz3OicnR7GxsdqzZ4+GDBnim3/LLbcoPj4+MB0CAJql63oPqKysTJIUFRXlNz83N1fR0dHq2bOn5s6dq3PnztW7jcrKSnm9Xr8JAND8NXgYdnV1tWbOnKlBgwapZ8+evvmPPPKIkpOTlZiYqH379umnP/2pCgsLtXbt2jq3k52drUWLFjW0DQBAE9XgAMrKytL+/fv14Ycf+s2fNm2a79+9evVSQkKChg8frqKiInXu3LnWdubOnavZs2f7Xnu9XiUlJTW0LQBAE9GgAJoxY4Y2bdqkrVu3qmPHjldcNz09XZJ06NChOgPI7XbL7XY3pA0AQBPmKICMMXr66ae1bt065efnKzU19ao1e/fulSQlJCQ0qEEAQPPkKICysrK0cuVKbdiwQRERESotLZUkeTwehYeHq6ioSCtXrtS9996r9u3ba9++fZo1a5aGDBmi3r17B+ULAAA0TY4CaNmyZZJq/tj0u1asWKEpU6YoLCxM77//vl599VVVVFQoKSlJEyZM0M9//vOANQwAaB4c34K7kqSkJBUUFFxXQwCAmwNPwwa+49NPP3VcM3z4cMc1X3/9teMaoLnhYaQAACsIIACAFQQQAMAKAggAYAUBBACwggACAFhBAAEArCCAAABWEEAAACsIIACAFQQQAMAKAggAYIXLXO0R1zeY1+uVx+Ox3QYA4DqVlZWpbdu29S7nCggAYAUBBACwggACAFhBAAEArCCAAABWEEAAACsIIACAFQQQAMAKAggAYAUBBACwggACAFjR6AKokT2aDgDQQFf7ed7oAqi8vNx2CwCAALjaz/NG9zTs6upqlZSUKCIiQi6Xy2+Z1+tVUlKSjhw5csUnrDZ3HIcaHIcaHIcaHIcajeE4GGNUXl6uxMREtWhR/3VOyxvY0zVp0aKFOnbseMV12rZte1OfYJdwHGpwHGpwHGpwHGrYPg7X8rE6je4WHADg5kAAAQCsaFIB5Ha7tWDBArndbtutWMVxqMFxqMFxqMFxqNGUjkOjG4QAALg5NKkrIABA80EAAQCsIIAAAFYQQAAAKwggAIAVTSaAlixZopSUFLVq1Urp6enatWuX7ZZuuIULF8rlcvlN3bt3t91W0G3dulVjxoxRYmKiXC6X1q9f77fcGKP58+crISFB4eHhyszM1MGDB+00G0RXOw5TpkypdX6MGjXKTrNBkp2drf79+ysiIkKxsbEaO3asCgsL/dY5f/68srKy1L59e7Vp00YTJkzQ8ePHLXUcHNdyHIYOHVrrfJg+fbqljuvWJALot7/9rWbPnq0FCxbo448/Vp8+fTRy5EidOHHCdms33G233aZjx475pg8//NB2S0FXUVGhPn36aMmSJXUuX7x4sV577TUtX75cO3fuVOvWrTVy5EidP3/+BncaXFc7DpI0atQov/Nj1apVN7DD4CsoKFBWVpZ27Nih9957TxcvXtSIESNUUVHhW2fWrFnauHGj1qxZo4KCApWUlGj8+PEWuw68azkOkjR16lS/82Hx4sWWOq6HaQIGDBhgsrKyfK+rqqpMYmKiyc7OttjVjbdgwQLTp08f221YJcmsW7fO97q6utrEx8ebl156yTfvzJkzxu12m1WrVlno8Ma4/DgYY8zkyZPNAw88YKUfW06cOGEkmYKCAmNMzfc+NDTUrFmzxrfO559/biSZ7du322oz6C4/DsYYc/fdd5sf//jH9pq6Bo3+CujChQvas2ePMjMzffNatGihzMxMbd++3WJndhw8eFCJiYlKS0vTo48+qsOHD9tuyari4mKVlpb6nR8ej0fp6ek35fmRn5+v2NhYdevWTU899ZROnz5tu6WgKisrkyRFRUVJkvbs2aOLFy/6nQ/du3dXp06dmvX5cPlxuCQ3N1fR0dHq2bOn5s6dq3Pnztlor16N7mnYlzt16pSqqqoUFxfnNz8uLk4HDhyw1JUd6enpysnJUbdu3XTs2DEtWrRIGRkZ2r9/vyIiImy3Z0Vpaakk1Xl+XFp2sxg1apTGjx+v1NRUFRUV6Wc/+5lGjx6t7du3KyQkxHZ7AVddXa2ZM2dq0KBB6tmzp6Sa8yEsLEyRkZF+6zbn86Gu4yBJjzzyiJKTk5WYmKh9+/bppz/9qQoLC7V27VqL3fpr9AGE/zd69Gjfv3v37q309HQlJyfr3Xff1RNPPGGxMzQGDz/8sO/fvXr1Uu/evdW5c2fl5+dr+PDhFjsLjqysLO3fv/+meB/0Suo7DtOmTfP9u1evXkpISNDw4cNVVFSkzp073+g269Tob8FFR0crJCSk1iiW48ePKz4+3lJXjUNkZKS6du2qQ4cO2W7FmkvnAOdHbWlpaYqOjm6W58eMGTO0adMmbdmyxe/zw+Lj43XhwgWdOXPGb/3mej7Udxzqkp6eLkmN6nxo9AEUFhamvn37Ki8vzzevurpaeXl5GjhwoMXO7Dt79qyKioqUkJBguxVrUlNTFR8f73d+eL1e7dy586Y/P44eParTp083q/PDGKMZM2Zo3bp1+uCDD5Samuq3vG/fvgoNDfU7HwoLC3X48OFmdT5c7TjUZe/evZLUuM4H26MgrsXq1auN2+02OTk55u9//7uZNm2aiYyMNKWlpbZbu6F+8pOfmPz8fFNcXGz+9re/mczMTBMdHW1OnDhhu7WgKi8vN5988on55JNPjCTzy1/+0nzyySfmq6++MsYY85//+Z8mMjLSbNiwwezbt8888MADJjU11XzzzTeWOw+sKx2H8vJy88wzz5jt27eb4uJi8/7775s77rjDdOnSxZw/f9526wHz1FNPGY/HY/Lz882xY8d807lz53zrTJ8+3XTq1Ml88MEHZvfu3WbgwIFm4MCBFrsOvKsdh0OHDpnnn3/e7N692xQXF5sNGzaYtLQ0M2TIEMud+2sSAWSMMa+//rrp1KmTCQsLMwMGDDA7duyw3dIN99BDD5mEhAQTFhZmOnToYB566CFz6NAh220F3ZYtW4ykWtPkyZONMTVDsefNm2fi4uKM2+02w4cPN4WFhXabDoIrHYdz586ZESNGmJiYGBMaGmqSk5PN1KlTm90vaXV9/ZLMihUrfOt888035oc//KFp166dueWWW8y4cePMsWPH7DUdBFc7DocPHzZDhgwxUVFRxu12m1tvvdXMmTPHlJWV2W38MnweEADAikb/HhAAoHkigAAAVhBAAAArCCAAgBUEEADACgIIAGAFAQQAsIIAAgBYQQABAKwggAAAVhBAAAAr/g9jGgX1BzXqKwAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ],
      "source": [
        "# Select a single image from the test dataset\n",
        "index = 0\n",
        "image = test_images[index]\n",
        "label = np.argmax(test_labels[index])\n",
        "\n",
        "# Make a prediction\n",
        "prediction = model.predict(np.expand_dims(image, axis=0))\n",
        "predicted_label = np.argmax(prediction)\n",
        "\n",
        "# Plot the image and show actual and predicted outputs\n",
        "plt.imshow(image.squeeze(), cmap='gray')\n",
        "plt.title(f'Actual: {label}, Predicted: {predicted_label}')\n",
        "plt.show()\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}