import matplotlib.pyplot as plt
import numpy as np

class Regressor:

  def __init__(self):
    self.x = []
    self.y = []
    self.m = 0
    self.b = 0

  def fit(self, x , y):
    self.x = x
    self.y = y
    # plt.scatter(x,y)
    # plt.xlabel("X axis")
    # plt.ylabel("Y axis")
    self.__compute_m_b()

  # Private Method
  def __sum(self):
    sum_x = 0
    for i in range(len(self.x)):
      sum_x = sum_x + self.x[i]

    return sum_x

  def __compute_summations(self):
    sum_x = sum(self.x)
    sum_y = sum(self.y)

    x_sqr = []
    for i in self.x:
      x_sqr.append(i * i)

    sum_x_sqr = sum(x_sqr)

    xy = []
    for i, j in zip(self.x, self.y):
      xy.append(i * j)

    sum_xy = sum(xy)

    return sum_x, sum_y, sum_x_sqr, sum_xy

  def __compute_m_b(self):
    sum_x, sum_y, sum_x_sqr, sum_xy = self.__compute_summations()
    self.m = ((len(self.x) * sum_xy) -
         (sum_x * sum_y)) / ((len(self.x) * sum_x_sqr) - (sum_x**2))
    self.b = (sum_y - (self.m * sum_x)) / len(self.x)

  def get_m_b(self):
    return self.m, self.b

  def predict(self, new_x):
    if type(new_x) is list:
      y_hat = []
      for i in new_x:
        y_hat.append((self.m * i) + self.b)
      return y_hat
    elif type(new_x) is int or float:
      return (self.m * new_x) + self.b
    else:
      return "Bad Data Type!!!!!"


def main():
  # x = np.array([1, 2, 3, 4, 5, 6])
  # y = np.array([1, 2, 3, 4, 5, 6])

  # reg = Regressor()
  # reg.fit(X, y)      # training_X, training_Y
  # print(reg.predict(7))
  x = np.array([1, 3, 3, 10, 2, 8])
  y = np.array([1, 0, 2, 5, 8, 12])

  reg = Regressor()
  reg.fit(x,y)
  m, b = reg.get_m_b()
  print(m, b)
  y_hat = reg.predict([1, 3, 5, 7, 9])


  plt.scatter(x,y, color="red")
  slope, intercept = np.polyfit(x,y,1)
  plt.plot(x, slope*x+intercept, color="blue")
  print("m: ", slope, "b: ", intercept)
  plt.show()

main()
