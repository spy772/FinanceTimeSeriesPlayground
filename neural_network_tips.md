General Neural Network Building Tips
====================================

Number of neurons per hidden layer
----------------------------------

These are general rules of thumb, but are not applicable to each situation. They do not take into account noise in the target values, the number of training cases, and the complexity (or lack thereof) of the function (source: http://www.faqs.org/faqs/ai-faq/neural-nets/part3/section-10.html). As such, these remain as rough starting points that work well overall, but can absolutely be fine-tuned. Fine-tuning is usually done by trying different amounts of neurons and/or layers and training models with these parameters until you reach desireable amounts (i.e. there is no known way to easily pick the values, but a good AI engineer will know how to better approximate given the data they have)

* The number of hidden neurons should be between the size of the input layer and the size of the output layer.
* The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
* The number of hidden neurons should be less than twice the size of the input layer.

A reminder that these are for the number of neurons **per hidden layer**
(source: https://www.heatonresearch.com/2017/06/01/hidden-layers.html, and above link). It is recommended that we increase the number of neurons in a hidden layer over adding another hidden layer if there seem to be issues (source: https://www.baeldung.com/cs/neural-networks-hidden-layers-criteria)


Number of hidden layers
-----------------------

Again, these are only guidelines and may not accurately represent the actual amount needed. (source for the entire section: https://www.baeldung.com/cs/neural-networks-hidden-layers-criteria)

* No hidden layers are needed when the data is linearly seperable
* 1 hidden layer is used when a problem can be continuously differentiable, and is non-linearly seperable (i.e. you cannot use a hyperplane to solve such problems)
    - An example is the XOR classification problem
* 2 hidden layers are used when you can represent a more arbitrary decision boundary
    - For example, when you can have solutions in a vector space where the target values are found in multiple clusters, and these clusters are not connected together (i.e. discontiguous regions)
    - Each neuron in the 2nd hidden layer learns each contiguous component (i.e. cluster) of the decision boundary
* 3 or more layers are required where there are patterns that become present over layers, rather than there being patterns over data (like in image recognition)
    - For example, in order to be able to tell the differenence between an image of a line and a circle, CNNs will have different weight matrices with each column corresponding to a portion of the image. In order to tell the difference between a line and circle, you will need to tell the difference between *each* column, which will require a layer to accurately "classify" that column before carrying on to the next layer's "classification"
    - As such, problems with higher levels of abstraction than any of the previously-mentioned layer counts will need 3+ hidden layers

As a general tip, it is recommended to keep the number of layers as low as possible (and needed). Most complex image recognition problems can be solved by using 8 hidden layers, and generating human-intelligible text can take about 96 layers. These are some of the most complex problems out there, and if we are using more than this, we are doing something wrong.

For our purposes, I believe that 1 or 2 hidden layers will suffice. If we fail to acquire desired training results with these amounts of layers, we most likely will need to process the data more finely (or get new data in general). Some things we could try are:
- Dimensionality reduction
- Standardization OR normalization (never do both at the same time)
- Adding a dropout layer (research about what these might be)


Choosing the right activation functions
---------------------------------------

Source: https://towardsdatascience.com/how-to-choose-the-right-activation-function-for-neural-networks-3941ff0e6f9c/

* Input layers don't need activation functions, they simply pass the data to the next layer
* Output layer activation function choice depends on the type of problem wanted to be solved
    - Regression can use identity (linear) activation functions, or any other relevant one that can provide a correctly-mapped value in the right number space
    - Binary classification can use sigmoid quite nicely
    - Multiclass classification can use softmax activation with the output layer having one node per possible class
    - Multilabel classification (can be assigned more than one class at a time) can use sigmoid with one node per possible class in the output layer
* Any non-linear activation function (this includes ReLU and any subcategories of ReLU, like LeakyReLU) can, and should, be used in hidden layers
    - Choice of function is chosen considering performance of the model or convergence of the loss function
* Some activation functions serve different types of deep learning models better
    - In MLP and CNN models, ReLU is a good default choice for hidden layers
    - In RNN, sigmoid or tanh (tanh typically has better performance) are good choices for hidden layers
    - Never use identity (purely linear) or softmax functions in hidden layers
    - Swish and HardSwish functions (derivations of exponential functions) are new types of activation functions in research, can test them out if available


Choosing the right optimizer
----------------------------

Here is an overview of the most popular optimizers (which try to minimize the loss/cost functions) and their uses. (source: https://musstafa0804.medium.com/optimizers-in-deep-learning-7bf81fed78a0)

* Gradient Descent
    - Iteratively reduces cost function by moving in the opposite direction of steepest ascent
    - Uses the entire data set in each iteration, which is computationally expensive
    - Pros:
        - Easy to understand and implement
    - Cons: 
        - Uses entire data set each iteration, super slow calculation and computation
        - Requires a large amount of memory as a result
* Stochastic Gradient Descent (SGD)
    - Varient of regular Gradient Descent, where it updates model parameters one-by-one
    - Pros: 
        - Frequently updates model params
        - Needs less memory
        - Allows for usage of large datasets and updates only one example at a time
    - Cons: 
        - Frequent updates can result in noisy and hard-to-converge gradients
        - High amount of variance in training, can get very different results each time
        - Frequent updates are computationally expensive
* Mini-Batch Gradient Descent
    - Combintation of SGD and batch gradient descent (never explained what this is, unless it is regular gradient descent which is quite possible)
        - Updates model params for each batch, and the dataset is split into numerous batches
    - Pros: 
        - More stable convergence than SGD
        - More efficient gradient calculations
        - Requires less memory
    - Cons:
        - Does not guarantee good convergence, still can have noisy convergence
* SGD with Momentum
    - Momentum is a way to nudge the gradient closer to 0 (a minima) by taking into account the previous step(s)' updates
    - Pros:
        - Momentum helps reduce noise
    - Cons:
        - Extra hyperparameter is added and needs to be optimized