import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

mnist_train = pd.read_csv('train.csv')

mnist_train_y = mnist_train['label']
mnist_train_x = mnist_train.drop(['label'], axis=1)

plt.imshow(mnist_train_x.iloc[3].values.reshape(28,28))
plt.show()