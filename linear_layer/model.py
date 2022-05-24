import numpy as np
from tqdm import trange

def cross_entropy_loss(x, y):
  b_out = np.zeros((y.shape[0], 10))
  b_out[range(y.shape[0]), y] = 1

  b_loss = -(b_out * x).sum(axis=1).mean()
  return b_loss, b_out

def relu_f(x):
  return x * (x > 0)

def sigmoid(x):
  return 1/(1+np.exp(-x))

class linear_layer():
  def __init__(self, bs):
    self.bs = bs
    self.input_w = np.random.uniform(-1., 1., size=(256, 28*28))/np.sqrt(256*28*28)
    self.hidden_w = np.random.uniform(-1., 1., size=(10, 256))/np.sqrt(256*10)
    self.x_1 = np.empty([self.bs, 28*28], dtype=np.float32)
    self.x_2 = np.empty([self.bs, 256], dtype=np.float32)
    self.x_relu = np.empty([self.bs, 256], dtype=np.float32)
    self.x_3 = np.empty([self.bs, 10], dtype=np.float32)
    self.x_lsm = np.empty([self.bs, 10], dtype=np.float32)

  def forward(self, input):
    self.x_1 = input.reshape(-1, 28*28)/255
    self.x_2 = self.input_w.dot(self.x_1.T)
    self.x_relu = relu_f(self.x_2)
    self.x_3 = self.hidden_w.dot(self.x_relu).T
    self.x_lsm = self.x_3 - np.log(np.sum(np.exp(self.x_3), axis=1)).reshape(-1, 1)

  def back_prop(self, lr, x_loss, y_one_hot):
    dl_dx_lsm = (-y_one_hot)/len(y_one_hot)
    dl_dx3 = dl_dx_lsm - np.exp(self.x_lsm)*dl_dx_lsm.sum(axis=1).reshape(-1, 1)
    dl_dw2 = self.x_relu.dot(dl_dx3)
    dl_dx_relu = dl_dx3.dot(self.hidden_w)
    dl_dx2 = (self.x_relu > 0).astype(np.float32) * dl_dx_relu.T
    dl_dw1 = dl_dx2.dot(self.x_1)

    self.input_w -= lr*dl_dw1
    self.hidden_w -= lr*dl_dw2.T

def train(model, x_train, y_train, iter, lr):
  losses, accuracies = list(), list()
  for i in (t := trange(iter)):
    samp = np.random.randint(0, 60000, size=(model.bs))
    x, y = x_train[samp], y_train[samp]
    model.forward(x)
    x_loss, y_one_hot = cross_entropy_loss(model.x_lsm, y)
    model.back_prop(lr, x_loss, y_one_hot)
    acc = (model.x_lsm.argmax(axis=1) == y).mean()
    accuracies.append(acc)
    t.set_description(f"Epoch: {i+1}")
    t.set_postfix(loss=x_loss, acc=acc)

  return losses, accuracies

def test(model, x_test, y_test):
  model.forward(x_test.reshape(-1, 28*28))
  print(f"Accuracy: {(model.x_lsm.argmax(axis=1).reshape(-1, 1) == y_test.reshape(-1, 1)).mean()}")
