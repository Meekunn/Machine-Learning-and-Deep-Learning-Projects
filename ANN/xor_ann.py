import numpy as np

class ANN:
  def __init__(self):
    np.random.seed(1)
    
    self.input_layer = np.array((2, 1))
    self.hidden_layer = np.array((2, 1))
    self.output_layer = np.array((1, 1))
    
    
    self.W_1 = np.random.random((2, 2))
    self.W_2 = np.random.random((2, 1))

    print("weight 1 is:", self.W_1)
    print("weight 2 is:", self.W_2)
    
    self.b_1 = np.ones((2, 1))
    self.b_2 = np.ones((1, 1))

    print("bias 1 is:", self.b_1)
    print("bias 2 is:", self.b_2)
    
    self.learning_rate = 0.1
    
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
    
  def sigmoid_derivative(self, x):
    return (self.sigmoid(x) * (1 - self.sigmoid(x)))

  def relu(self, x):
    print(np.maximum(0, x))
    return np.maximum(0, x)
    
  def relu_derivative(self, x):
    if x > 0:
      return x.fill(1)
    else:
      return x * 0

  def loss_derivative(self, y_hat, ground_truth):
    return -2 * (ground_truth - y_hat)
      
  def forward_propagation(self, input_data):
    input_data = input_data.reshape((2, 1))
    print("input data is:", input_data)
    
    z_1 = np.matmul(self.W_1.T, input_data) + self.b_1
    a_1 = self.relu(z_1)

    print("z_1 is:", z_1)
    z_2 = np.matmul(self.W_2.T, a_1) + self.b_2
    a_2 = self.sigmoid(z_2)
    return a_2

  def back_propagation(self, input_data, target_data, output_data):

    h_1 = np.matmul(self.W_1.T, input_data) + self.b_1
    c_1 = np.matmul(self.W_2.T, h_1) + self.b_2
    dy_hat_dW2 = h_1 * self.sigmoid_derivative(c_1)

    dL_dW2 = self.loss_derivative(output_data, target_data) * dy_hat_dW2
    W_2_new = self.W_2.T - (self.learning_rate * dL_dW2)

    c_2 = np.matmul(self.W_1.T, input_data) + self.b_1
    dy_hat_dh = self.W_2.T * c_1
    dh_dW_1 = np.matmul(input_data, self.relu_derivative(c_2))
    
    dL_dW1 = self.loss_derivative(output_data, target_data) * dy_hat_dh * dh_dW_1
    W_1_new = self.W_1 - (self.learning_rate * dL_dW1)
    print(W_1_new)
    print(W_2_new)

def main():
  ann = ANN()
  input_data = np.array((2, 1), dtype=np.int32)
  input_data[0] = 0
  input_data[1] = 1
  target_data = [1]
  
  output = ann.forward_propagation(input_data)
  print(output)
  ann.back_propagation(input_data,target_data, output )

main()