import pandas as pd
from torch.utils.data import DataLoader

def load_dataset():
  train_data = pd.read_csv(r"data\mnist_train.csv")
  test_data = pd.read_csv(r"data\mnist_test.csv")

  train, test = train_data.values, test_data.values

  y_train, x_train = train[:, 0], train[:, 1:].reshape(-1, 28, 28)
  y_test, x_test = test[:, 0], test[:, 1:].reshape(-1, 28, 28)

  return x_train, y_train, x_test, y_test

if __name__ == '__main__':
  train_dataloader, test_dataloader = load_dataset()
