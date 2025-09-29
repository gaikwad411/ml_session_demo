
"""
Assignment 2 - Neural Networks
Name: Sachin Vilas Gaikwad
Enroll.No.: MT24AAI195

1. Implement vanilla gradient descent algorithm for sigmoid neuron with
   input output dataset consisting two samples: { (0.5, 0.2), (2.4, 0.9) }
   Clearly show the error surface and evolution of parameters with every
   iteration of vanilla GD.

Valilla Gradient Descent algorithm implmentation.
"""
#####################################################################
# Import required python packages
#####################################################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
from typing import List

#####################################################################
# User defined functions
#####################################################################


def yf(w: float, x:float, b:float) -> float:
  """
  Get the lienar equation value w*x+b

  Args:
    w (float): Weight parameter value
    b (float): bias parameter value
    x (float): x - input feature value
  
  Returns:
    (float): Output of linear equation value.
  """
  return w * x + b


def f(x:float , w:float, b:float) -> float:
  """
  Get the Y function value
  
  Args:
    w (float): Weight parameter value
    b (float): bias parameter value
    x (float): x - input feature value
  
  Returns:
    (float): Output of sigmoid value for given data input x, weight w and bias b.
  """
  return 1.0 / (1.0 + np.exp(-yf(w, x, b)))


def get_grad_W(w:float, b:float, x:float, y:float) -> float:
  """
  Get gradient of loss function value with respect to weight w.

  Args:
    w (float): Weight parameter value
    b (float): bias parameter value
    x (float): x - input feature value
    y (float): y - output value - the truth value for input x
  
  Returns:
    (float): Output of gradident of loss function value with respect to weight w.
  """
  Fx = f(x, w, b)
  return ( (Fx-y) * Fx * (1-Fx) * x )


def get_grad_b(w:float, b:float, x:float, y:float) -> float:
  """
  Get gradient of loss function value with respect to bias b.

  Args:
    w (float): Weight parameter value
    b (float): bias parameter value
    x (float): x - input feature value
    y (float): y - output value - the truth value for input x
  
  Returns:
    (float): Output of gradident of loss function value with respect to bias b.
  """
  Fx = f(x, w, b)
  return ( (Fx-y) * Fx * (1-Fx) )


def compute_loss(w:float, b:float, x:float, y:float) -> float:
  """
  Define prediction and loss (MSE)

  Args:
    w (float): Weight parameter value
    b (float): bias parameter value
    x (float): x - input feature value
    y (float): y - output value - the truth value for input x
  
  Returns:
    (float): Computed loss function value for given w, b, x and y.

  """
  y_pred = f(x, w, b)
  return np.mean((y - y_pred)**2)


def get_accuracy(y:float, y_hat:float) -> float:
  """
  Get accuracy from given y (truth) and y_hat (predicted)

  Args:
    y (float): Truth value from dataset of target variable.
    y_hat (float) : Predicted value 
  
  Returns:
    (float): Accuracy of truth w.r.t. predicted value.
  """
  return  (y_hat / y) * 100


def plot_loss_over_iterations(iterations: List[int], 
                              loss_history: List[float],
                              title: str = '') -> None:
  """
  Plot the loss over iterations

  Args:
    iterations(List[int]): List of iteration counts.
    loss_history(List[float]): List of loss accumulated over iterations.
    
  Returns:
    (None)
  """
  if not title:
    title = 'Loss over Iterations (Sigmoid Neuron)'
  plt.plot(iterations, loss_history)
  plt.xlabel('Iterations')
  plt.ylabel('Loss')
  plt.title(title)
  plt.show()


def plot_error_surface(weights_history: List[float], 
                       bias_history: List[float], 
                       loss_history: List[float], 
                       train_path_title:str = '') -> None:
  """
  Plot the error surface for given weights, bias and loss
  history.

  Args:
    weights_history (List[float]): List of weights accumulated over iterations.
    bias_history (List[float]): List of bias accumulated over iterations.
    loss_history (List[float]): List of loss accumulated over iterations.
  
  Returns:
    (None)

  """
  if not train_path_title:
    train_path_title = "Training path"
  # Create grid of W and b values
  W_vals = np.linspace(-3, 3, 50)   # adjust ranges as needed
  b_vals = np.linspace(-3, 3, 50)
  W_grid, B_grid = np.meshgrid(W_vals, b_vals)

  # Compute loss surface
  Loss_grid = np.zeros_like(W_grid)
  for i in range(W_grid.shape[0]):
      for j in range(W_grid.shape[1]):
          Loss_grid[i, j] = compute_loss(W_grid[i, j], B_grid[i, j], X, Y)

  # Plot surface
  fig = plt.figure(figsize=(10, 7))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(W_grid, B_grid, Loss_grid, cmap='viridis', alpha=0.8)

  # Add color bar
  fig.colorbar(surf, shrink=0.5, aspect=5, label="Loss")

  # Labels
  ax.set_xlabel("Weight (W)")
  ax.set_ylabel("Bias (b)")
  ax.set_zlabel("Loss")
  ax.plot(weights_history, bias_history, loss_history, 
          color="red", marker="o", label=train_path_title)
  ax.legend()
  plt.show()


def grad_descent_alg_execute(X: np.array, Y: np.array, 
                             step_size: float = 0.1, 
                             max_iteration: int = 7000,
                             w_initial: float | None = None,
                             b_initial: float | None = None
                             )-> tuple:
  """
  Execute Gradient Descent Algorithm for given X, Y and step size. Get the 
  weight and bias required by output function f to predict.

  Args:
    X (np.array): array (shape n x 1) of input data points.
    Y (np.array): array (shape n x 1) of output data points.
    step_size (float): Step size in Gradient descent algorithm.
    max_iteration (float): Total iterations for learning parameters

  Returns:
    (tuple): tuple of result items, 
    1. float - learned weight parameter
    2. float - learned bias parameter
    3. list[float] - list of loss values accumulated over iterations.
    4. list[int] - list of iteration counts accumulated over iterations.
    5. list(float) - list of learned weight values accumulated over iterations.
    6. list(float) - list of learned bias values accumulated over iterations.
  """  
  loss_history = []
  iterations = []
  weights_history = []
  bias_history = []

  # Initialize weight and bias randomly
  w = np.random.rand()
  if w_initial is not None:
    w = w_initial

  b = np.random.rand()
  if b_initial:
    b = b_initial

  # Set current iteration number
  current_iteration = 0
  # Until max iterations
  # 1. For each data point calculate gradident of loss function with respect
  #     to weight, bias and loss
  # 2. With step size calculate value of weight and bias for the iteration.
  # 3. Accumulate list of weights, bias, iterations, loss over iterations.
  while current_iteration < max_iteration:    
    dw = 0
    db = 0
    loss = 0
    # For every data point
    for data_index, x in enumerate(X):    
      # Get sum of gradient w.r.t weight  
      dw = dw + get_grad_W(w, b, X[data_index], Y[data_index])
      # Get sum of gradient w.r.t. bias
      db = db + get_grad_b(w, b, X[data_index], Y[data_index])                  
      loss = loss + np.mean((Y[data_index] - f(x, w, b))**2)

    # Get the new weight value 
    w = w - step_size * dw

    # Get the new bias value
    b = b - step_size * db
    
    # Increment the iterations
    current_iteration += 1

    # For every 100th iteration, record loss, weights and bias values.
    if current_iteration % 100 == 0:
      loss_history.append(loss)
      iterations.append(current_iteration)
      weights_history.append(w)
      bias_history.append(b)
  
  print(f"Parameters learned are W: {round(w, 4)} and b: {round(b, 4)}")
  print(f"Step size: {step_size} Loss:{loss}")
  return w, b, loss_history, iterations, weights_history, bias_history

#####################################################################
# Execution
#####################################################################
# Input 
X = np.array([0.5, 2.4])

# Output
Y = np.array([0.2, 0.9])

# Execute gradient descent algorithm
w, b, loss_history, iterations, weights_history, bias_history \
= grad_descent_alg_execute(X, Y)

# Plot the loss over iterations
plot_loss_over_iterations(iterations, loss_history)

# Plot the error surface for given weights, bias and loss history.
plot_error_surface(weights_history, bias_history, loss_history)

# Print the data and predicted values as per learned parameters
for i, x in enumerate(X):
  y_hat = f(X[i], w, b)
  y = Y[i]
  accuracy = get_accuracy(y, y_hat)  
  print(f"y:{y} y_hat:{round(y_hat, 4)} Accuracy: {round(accuracy, 2)}")

"""
2. Provide your observations in 4-5 statements to the evolution of GD for
   gentle slope regions and steep slope regions on the error surface. 
"""

# Execute gradient descent algorithm, with step size for gentle slope 
w, b, loss_history, iterations, weights_history, bias_history \
= grad_descent_alg_execute(X, Y, w_initial=0, b_initial=0)


# Plot the error surface for given weights, bias and loss history.
plot_error_surface(weights_history, bias_history, loss_history,
                          train_path_title="Gentle slope path")

# Execute gradient descent algorithm, with step size for steep slope 
w, b, loss_history, iterations, weights_history, bias_history \
= grad_descent_alg_execute(X, Y, w_initial=-2, b_initial=2.5)


# Plot the error surface for given weights, bias and loss history.
plot_error_surface(weights_history, bias_history, loss_history,
                          train_path_title="Steep slope path")

"""
3. Is there any effect of step size on the progress of GD in minimization of loss function.
   What happens to GD when size is large. What happens to GD when the step size is negative.
   Provide your justificaiton by simulation. 
"""
# Execute gradient descent algorithm, with step size for large step size 
w, b, loss_history, iterations, weights_history, bias_history \
= grad_descent_alg_execute(X, Y, step_size=0.5, w_initial=-2, b_initial=2.5)


# Plot the error surface for given weights, bias and loss history.
plot_error_surface(weights_history, bias_history, loss_history,
                          train_path_title="Large step size path")

# Execute gradient descent algorithm, with step size for negative step size 
w, b, loss_history, iterations, weights_history, bias_history \
= grad_descent_alg_execute(X, Y, step_size=-0.1, w_initial=-2, b_initial=2.5)


# Plot the error surface for given weights, bias and loss history.
plot_error_surface(weights_history, bias_history, loss_history,
                          train_path_title="Negative step size path")
