Download Link: https://assignmentchef.com/product/solved-csci5561-homework-4-neutral-network
<br>
Figure 1: You will implement (1) a multi-layer perceptron (neural network) and (2) convolutiona neural network to recognize hand-written digit using the MNIST dataset.

The goal of this assignment is to implement neural network to recognize hand-written digits in the MNIST data.

<strong>MNIST Data </strong>You will use the MNIST hand written digit dataset to perform the first task (neural network). We reduce the image size (28 × 28 → 14 × 14) and subsample the data. You can download the training and testing data from here: <a href="http://www.cs.umn.edu/~hspark/csci5561/ReducedMNIST.zip">http://www.cs.umn.edu/</a><a href="http://www.cs.umn.edu/~hspark/csci5561/ReducedMNIST.zip">~</a><a href="http://www.cs.umn.edu/~hspark/csci5561/ReducedMNIST.zip">hspark/csci5561/ReducedMNIST.zip</a>

<em>Description</em>: The zip file includes two MAT files (mnist_train.mat and mnist_test.mat). Each file includes im_* and label_* variables:

<ul>

 <li>im_* is a matrix (196 × <em>n</em>) storing vectorized image data (196 = 14 × 14)</li>

 <li>label_* is <em>n </em>× 1 vector storing the label for each image data.</li>

</ul>

<em>n </em>is the number of images. You can visualize the <em>i</em><sup>th </sup>image, e.g., imshow(uint8(reshape(im_train(:,i), [14,14]))).

<h1>2             Single-layer Linear Perceptron</h1>

Figure 2: You will implement a single linear perceptron that produces accuracy near 30%. Random chance is 10% on testing data.

You will implement a single-layer <em>linear </em>perceptron (Figure 2(a)) with stochastic gradient descent method. We provide main_slp_linear where you will implement GetMiniBatch and TrainSLP_linear.

function [mini_batch_x, mini_batch_y] = GetMiniBatch(im_train, label_train, batch_size)

<strong>Input: </strong>im_train and label_train are a set of images and labels, and batch_size is the size of the mini-batch for stochastic gradient descent.

<strong>Output: </strong>mini_batch_x and mini_batch_y are cells that contain a set of batches (images and labels, respectively). Each batch of images is a matrix with size 194×batch_size, and each batch of labels is a matrix with size 10×batch_size (one-hot encoding). Note that the number of images in the last batch may be smaller than batch_size. <strong>Description: </strong>You may randomly permute the the order of images when building the batch, and whole sets of mini_batch_* must span all training data.

function y = FC(x, w, b)

<strong>Input: </strong>x∈R<em><sup>m </sup></em>is the input to the fully connected layer, and w∈R<em><sup>n</sup></em><sup>×<em>m </em></sup>and b∈R<em><sup>n </sup></em>are the weights and bias.

<strong>Output: </strong>y∈R<em><sup>n </sup></em>is the output of the linear transform (fully connected layer). <strong>Description: </strong>FC is a linear transform of <strong>x</strong>, i.e., <strong>y </strong>= <strong>wx </strong>+ <strong>b</strong>.

function [dLdx dLdw dLdb] = FC_backward(dLdy, x, w, b, y)

<strong>Input: </strong>dLdy∈R<sup>1×<em>n </em></sup>is the loss derivative with respect to the output <strong>y</strong>.

<strong>Output: </strong>dLdx∈R<sup>1×<em>m </em></sup>is the loss derivative with respect the input <strong>x</strong>, dLdw∈R<sup>1×(<em>n</em>×<em>m</em>) </sup>is the loss derivative with respect to the weights, and dLdb∈R<sup>1×<em>n </em></sup>is the loss derivative with respec to the bias.

<strong>Description: </strong>The partial derivatives w.r.t. input, weights, and bias will be computed. dLdx will be back-propagated, and dLdw and dLdb will be used to update the weights and bias.

function [L, dLdy] = Loss_euclidean(y_tilde, y)

<strong>Input: </strong>y_tilde∈R<em><sup>m </sup></em>is the prediction, and y∈ 0<em>,</em>1<em><sup>m </sup></em>is the ground truth label.

<strong>Output: </strong>L∈R is the loss, and dLdy is the loss derivative with respect to the prediction. <strong>Description: </strong>Loss_euclidean measure Euclidean distance <em>L </em>= k<strong>y </strong>− <strong>y</strong>k<sup>2</sup>. e

function [w, b] = TrainSLP_linear(mini_batch_x, mini_batch_y)

<strong>Input: </strong>mini_batch_x and mini_batch_y are cells where each cell is a batch of images and labels.

<strong>Output: </strong>w∈R<sup>10×196 </sup>and b∈R<sup>10×1 </sup>are the trained weights and bias of a single-layer perceptron.

<strong>Description: </strong>You will use FC, FC_backward, and Loss_euclidean to train a singlelayer perceptron using a stochastic gradient descent method where a pseudo-code can be found below. Through training, you are expected to see reduction of loss as shown in Figure 2(b). As a result of training, the network should produce more than 25% of accuracy on the testing data (Figure 2(c)).

<strong>Algorithm 1 </strong>Stochastic Gradient Descent based Training

1: Set the learning rate <em>γ</em>

2: Set the decay rate <em>λ </em>∈ (0<em>,</em>1]

3: Initialize the weights with a Gaussian noise <strong>w </strong>∈N(0<em>,</em>1)

4: <em>k </em>= 1

5: <strong>for </strong>iIter = 1 : nIters <strong>do</strong>

6:               At every 1000<sup>th </sup>iteration, <em>γ </em>← <em>λγ</em>

7:              0 and

8:                <strong>for </strong>Each image <strong>x</strong><em><sub>i </sub></em>in <em>k</em><sup>th </sup>mini-batch <strong>do</strong>

9:                     Label prediction of <strong>x</strong><em><sub>i</sub></em>

10:                   Loss computation <em>l</em>

11:                  Gradient back-propagation of  using back-propagation.

12:                  <strong>w                             w                       </strong>and

13:           <strong>end for</strong>

14:              <em>k</em>++ (Set <em>k </em>= 1 if <em>k </em>is greater than the number of mini-batches.)

15:

<h1>3             Single-layer Perceptron</h1>

Figure 3: You will implement a single perceptron that produces accuracy near 90% on testing data.

You will implement a single-layer perceptron with <em>soft-max cross-entropy </em>using stochastic gradient descent method. We provide main_slp where you will implement TrainSLP. Unlike the single-layer linear perceptron, it has a soft-max layer that approximates a max function by clamping the output to [0<em>,</em>1] range as shown in Figure 3(a).

function [L, dLdy] = Loss_cross_entropy_softmax(x, y)

<strong>Input: </strong>x∈R<em><sup>m </sup></em>is the input to the soft-max, and y∈ 0<em>,</em>1<em><sup>m </sup></em>is the ground truth label.

<strong>Output: </strong>L∈R is the loss, and dLdy is the loss derivative with respect to <strong>x</strong>.

<strong>Description: </strong>Loss_cross_entropy_softmax measure cross-entropy between two distributionswhere <strong>y</strong>e<em><sub>i </sub></em>is the soft-max output that approximates the max operation by clamping <strong>x </strong>to [0<em>,</em>1] range:

<em>,</em>

where <strong>x</strong><em><sub>i </sub></em>is the <em>i</em><sup>th </sup>element of <strong>x</strong>.

function [w, b] = TrainSLP(mini_batch_x, mini_batch_y)

<strong>Output: </strong>w∈R<sup>10×196 </sup>and b∈R<sup>10×1 </sup>are the trained weights and bias of a single-layer perceptron.

<strong>Description: </strong>You will use the following functions to train a single-layer perceptron using a stochastic gradient descent method: FC, FC_backward, Loss_cross_entropy_softmax

Through training, you are expected to see reduction of loss as shown in Figure 3(b). As a result of training, the network should produce more than 85% of accuracy on the testing data (Figure 3(c)).

<h1>4             Multi-layer Perceptron</h1>

Figure 4: You will implement a multi-layer perceptron that produces accuracy more than 90% on testing data.

You will implement a multi-layer perceptron with a single hidden layer using a stochastic gradient descent method. We provide main_mlp. The hidden layer is composed of 30 units as shown in Figure 4(a).

function [y] = ReLu(x)

<strong>Input: </strong>x is a general tensor, matrix, and vector.

<strong>Output: </strong>y is the output of the Rectified Linear Unit (ReLu) with the same input size. <strong>Description: </strong>ReLu is an activation unit (<strong>y</strong><em><sub>i </sub></em>= max(0<em>,</em><strong>x</strong><em><sub>i</sub></em>)). In some case, it is possible to use a Leaky ReLu (<strong>y</strong>) where

function [dLdx] = ReLu_backward(dLdy, x, y)

<strong>Input: </strong>dLdy∈R<sup>1×<em>z </em></sup>is the loss derivative with respect to the output <strong>y </strong>∈R<em><sup>z </sup></em>where <em>z </em>is the size of input (it can be tensor, matrix, and vector).

<strong>Output: </strong>dLdx∈R<sup>1×<em>z </em></sup>is the loss derivative with respect to the input <strong>x</strong>.

function [w1, b1, w2, b2] = TrainMLP(mini_batch_x, mini_batch_y)

<strong>Output: </strong>w1 ∈R<sup>30×196</sup>, b1 ∈R<sup>30×1</sup>, w2 ∈R<sup>10×30</sup>, b2 ∈R<sup>10×1 </sup>are the trained weights and biases of a multi-layer perceptron.

<strong>Description: </strong>You will use the following functions to train a multi-layer perceptron using a stochastic gradient descent method: FC, FC_backward, ReLu, ReLu_backward, Loss_cross_entropy_softmax. As a result of training, the network should produce more than 90% of accuracy on the testing data (Figure 4(b)).

<h1>5             Convolutional Neural Network</h1>

Input        Conv (3)        ReLu              Pool (2×2)      Flatten   FC   Soft-max                              Accuracy: 0.947251

(a) CNN                                                                              (b) Confusion

Figure 5: You will implement a convolutional neural network that produces accuracy more than 92% on testing data.

You will implement a convolutional neural network (CNN) using a stochastic gradient descent method. We provide main_cnn. As shown in Figure 4(a), the network is composed of: a single channel input (14×14×1) → Conv layer (3×3 convolution with 3 channel output and stride 1) → ReLu layer → Max-pooling layer (2 × 2 with stride 2) → Flattening layer (147 units) → FC layer (10 units) → Soft-max. function [y] = Conv(x, w_conv, b_conv)

<strong>Input: </strong>x∈R<em><sup>H</sup></em><sup>×<em>W</em>×<em>C</em></sup><sup>1 </sup>is an input to the convolutional operation, w_conv∈R<em><sup>H</sup></em><sup>×<em>W</em>×<em>C</em></sup><sup>1×<em>C</em></sup><sup>2 </sup>and b_conv∈R<em><sup>C</sup></em><sup>2 </sup>are weights and bias of the convolutional operation.

<strong>Output: </strong>y∈R<em><sup>H</sup></em><sup>×<em>W</em>×<em>C</em></sup><sup>2 </sup>is the output of the convolutional operation. Note that to get the same size with the input, you may pad zero at the boundary of the input image. <strong>Description: </strong>This convolutional operation can be simplified using MATLAB built-in function im2col.

function [dLdw, dLdb] = Conv_backward(dLdy, x, w_conv, b_conv, y) <strong>Input: </strong>dLdy is the loss derivative with respec to <strong>y</strong>.

<strong>Output: </strong>dLdw and dLdb are the loss derivatives with respect to convolutional weights and bias <strong>w </strong>and <strong>b</strong>, respectively.

<strong>Description: </strong>This convolutional operation can be simplified using MATLAB built-in function im2col. Note that for the single convolutional layer, is not needed.

function [y] = Pool2x2(x)

<strong>Input: </strong>x∈R<em><sup>H</sup></em><sup>×<em>W</em>×<em>C </em></sup>is a general tensor and matrix.

<strong>Output: </strong>y∈R<em><u><sup>H</sup></u></em><sub>2 </sub><sup>×</sup><em><u><sup>W</sup></u></em><sub>2 </sub><sup>×<em>C </em></sup>is the output of the 2×2 max-pooling operation with stride 2. function [dLdx] = Pool2x2_backward(dLdy, x, y) <strong>Input: </strong>dLdy is the loss derivative with respect to the output <strong>y</strong>. <strong>Output: </strong>dLdx is the loss derivative with respect to the input <strong>x</strong>.

function [y] = Flattening(x) <strong>Input: </strong>x∈R<em><sup>H</sup></em><sup>×<em>W</em>×<em>C </em></sup>is a tensor.

<strong>Output: </strong>y∈R<em><sup>HWC </sup></em>is the vectorized tensor (column major).

function [dLdx] = Flattening_backward(dLdy, x, y) <strong>Input: </strong>dLdy is the loss derivative with respect to the output <strong>y</strong>. <strong>Output: </strong>dLdx is the loss derivative with respect to the input <strong>x</strong>.

function [w_conv, b_conv, w_fc, b_fc] = TrainCNN(mini_batch_x, mini_batch_y) <strong>Output: </strong>w_conv ∈ R<sup>3×3×1×3</sup>, b_conv ∈ R<sup>3</sup>, w_fc ∈ R<sup>10×147</sup>, b_fc ∈ R<sup>147 </sup>are the trained weights and biases of the CNN.

<strong>Description: </strong>You will use the following functions to train a convolutional neural network using a stochastic gradient descent method: Conv, Conv_backward, Pool2x2,

Pool2x2_backward, Flattening, Flattening_backward, FC, FC_backward, ReLu, ReLu_backward, Loss_cross_entropy_softmax. As a result of training, the network should produce more than 92% of accuracy on the testing data (Figure 5(b)).