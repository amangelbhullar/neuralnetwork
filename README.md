# neuralnetwork

## generating vector x

We are using python3 on jupyter notebook to implement question 1. First, we create a vector x, containing 5000 observations drawn from a normal distribution with mean 0 and variance 1. This will represent a vector of a set of x inputs. 

## generating vector eps

Secondly, we are creating vector eps containing 5000 observations drawn from a normal distribution with mean 0 and variance 0.25.

## generating y from x and eps using model 

Now, we are generating vector y according to model y = -1 + 0.5x – 2x^2 + 0.3x^3 + eps using x and eps vectors. We are storing these vector y points in variable y and writing them in a file.

Here in Q1(d), we are implementing Adaline (Adaptive neuron) neuron learning algorithm. We Initially started weights to be very small numbers (i.e., close to 0) and for each training session we computed the prediction and then we updated the weights. We updated the weights ∆𝑊 = 𝑥𝜆 (𝑦−y^) (delw variable in code for ∆𝑊). In Stochastic/Iterative gradient descent, we input one training example at a time and carry out forward and back propagation to update the weights.  However, In Batch gradient descent, we input the entire training dataset and then carry out forward and back propagation. The perceptron will fire if the weighted sum of its inputs is greater than threshold. There will always be a sudden change in the decision from 0 to 1 when ∑_(i=1)^n▒〖w_i x_i 〗 crosses the threshold. 

To smooth this sudden change in the decision (i.e. 0 or 1), we are using the sigmoid function. The sigmoid function is a family of functions and one of them is called the logistic function. 
 
Logistic function = 1/(1+e^(-(w_0  ∑_(i=1)^n▒〖w_i x_i 〗)) ) , 

degree d = 3, Mean Square Error calculated with Batch Gradient Descent, stochastic/Iterative Gradient Descent and logistic sigmoid function. As we observe from the table below, the mean square error is least with Logistic sigmoid function compared to Batch Gradient Descent, stochastic/Iterative Gradient Descent.![image]

