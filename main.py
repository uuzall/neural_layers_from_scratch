import matplotlib.pyplot as plt
import data_loading
import model
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', help='Learning Rate', type=float, default=0.1)
  parser.add_argument('--bs', help='Batch Size', type=int, default=64)
  parser.add_argument('--iter', help='Number of Iterations', type=int, default=10000)
  args = parser.parse_args()

  print('Loading the Dataset.')
  x_train, y_train, x_test, y_test = data_loading.load_dataset()
  print('Dataset Successfully Loaded.')
  print()
  net = model.linear_layer(args.bs)
  print('Training Network')
  losses, accuracies = model.train(net, x_train, y_train, args.iter, args.lr)
  print('Network Successfully Trained')
  print()
  print('Testing Network')
  model.test(net, x_test, y_test)

  plt.plot(losses)
  plt.plot(accuracies)
  plt.ylim(0, 1.1)
  plt.show()
  print('Fin.')
