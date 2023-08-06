# handwritten_digit_recognition_from_scratch
in this repository i tried to solve handwritten digits recognition problem with different nueral network architecture.

- without using common NNs libraries like tensorflow or pytorch just with pure code
- mnist dataset has been used
- gradient descent and backpropagation method have been used

## dumbest network
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mehrdadghassabi/Gracc/blob/master/Gracc.ipynb) <br />
as mnist dataset pictures are $`28 \times 28`$ the input layer would have 784 nodes and the output layer would have 10 nodes each one corresponds to probablity of being a digit,
the dumbest network for such a task is fully connecting the input and the output layer without any hidden layer.<br />
### training
as there is no hidden layer in this network only two vector would determine the answer a $`784 \times 10`$ weight vector and a $`10 \times 1`$ bias vector,
the output would be $`A1 = W^T \times A0 + B`$ , 
where W is weight vector B is bias vector A0 is flatten version of the grey scale input image (a $`784 \times 1`$ vector). <br />
according to gradient descent algorithm first of all W and B should be initialized randomly and get updated to minimize the cost function,
where $`Cost = (y - \hat{y})^2`$ and update rules are


![diag](https://github.com/Mehrdadghassabi/handwritten_digit_recognition_from_scratch/assets/53050138/fb1743a7-9bf5-49c9-b61e-896486f696b2)
