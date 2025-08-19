Building Forecasting Neural Networks Summary
============================================

Types of architectures
----------------------

There are numerous neural network architectures, and the majority of them can work for time series forecasting. Of course, some better than others.  

* Pure deep learning (dense layers): 
    - Using the most basic form of neural networks, each unit in a dense layer is simply an artificial neuron
    - This is great if you want a lower-compute model to be built, but the tradeoff is that it is not able to as easily generalize as other architectures might
    - It will have a hard time with noisy data, and even with recognizing that there are trends in the model
* RNNs (Reccurent Neural Networks)
    - A step in a better direction compared to pure dense-layer networks, due to the hidden state (only in a stateful RNN, which is most of them these days; and this state serves as "memory")
    - This hidden state (memory) only holds a few prior steps of processing (in our case, steps would be windows processed in a time-series)
    - With each new input processed during training, the RNN takes into account the current hidden state as context, to see if there is any sequential meaning to the current input compared to the previous inputs
    - The hidden state is updated with every input, usually only remembering inputs that have a strong sequential correlation
    - However, this state is quite short-term, and back-propagation through an RNN with long sequences can have what's called the "exploding gradient problem" where it is impossible to calculate the gradient appropriately since the context further past a certain point in the hidden state is lost
    - RNNs struggle with dealing with long sequences of data (theoretically they can work quite well on shorter sequences)
    - RNNs also tend to use the `tanh` activation function as default, and it is discouraged to use `ReLU` since ReLU is very susceptible to the "exploding/vanishing" gradient problem (need to research more as to why this is)
* LSTMs (Long Short Term Memory) 
    - To mitigate the issues of RNNs, LSTMs are used as they incoroporate the RNN structure into each cell/unit of an LSTM model
    - LSTMs have "forget gates" which allow for the hidden state to be reset and updated appropriately
        - As an example, an LSTM cell might have a pattern stored in the hidden state where it identified a large spike in the data set, but it just ran across a part in the time series where there's a sudden decrease
        - The forget gate is likely to forget the upward spike and replace it with the sudden decrease, so that the model knows it is in a new portion of the time series data
    - LSTMs are much better at dealing with longer sequences of data, but still maintain the RNNs capability at dealing with short-term data well too
        - They are more computationally heavy than RNNs since they have more computations done in each cell compared to RNNs
    - LSTMs have limitations of course, they still cannot handle immensely long time series well, but they are much better than RNNs at handling longer time series
        - One way you can prepare longer timeseries for use with LSTMs is by using CNNs as a form of pre-processing
* CNNs (Convolutional Neural Networks)
    - They are exceptionally good at image recognition tasks, but they are also good at being able to identify patterns in time series data as well
    - The way they work is by processing certain sequences (windows) of the time series and making predictions based on specific filters (kernels) applied to these sequences
        - For example, in images you typically use 2D CNN layers (one for the x and y axes respectively) and you set an appropriate amount of kernels, where they will attempt to learn texture, another edges, another shapes, etc.
        - For our use case, we can try experimenting with a different amount of kernels, where the kernels learn different patterns (like local trends, local spikes, etc.)
        - There doesn't seem to be a strict method on picking the amounts of kernels, because it isn't necessarily a correlation of 1 kernel = 1 aspect (CNNs don't understand trends, spikes, and other patterns as an aspect, they just mathematically attempt to find relationships arbitrarily)
        - Kernels are simply a linear combination of the time/data points (in that kernel) and the weights, hence there not being a strict correlation between a kernel and an "aspect" we might see with our eyes
        - You set the number of kernels a CNN layer should have, and also the amount of data points (the kernel size) each kernel should filter
    - CNNs are great for learning short-term patterns and trends, so they can be coupled alongside LSTMs to make good predictions
    - You can also use a 1D CNN layer on a time series as a form of pre-processing  
        - This works by the CNN extracting high-level features like local trends or spikes, and then LSTMs analyzing the extracted sequences to learn longer-term dependencies
    - The difference between CNNs and RNNs is that CNNs are better for local pattern recognition, while RNNs are better for sequential dependency recognition amongst data
    - Another common way of using CNNs is to build a few layers of them, and incidentally research finds that earlier-layer CNNs tend to be good at extracting high-level features, and then the subsequent layers sequentially extract lower and lower level features


## Choosing TensorFlow instead of aeon

The choice was made to use TensorFlow over aeon for time series forecasting, mainly due to the fact that aeon cannot handle multivariate (and thus multiple time series) time series inputs, and a custom solution had to be created. Overall, it will likely perform better anyways since we may hypertune the model according to this problem's specific needs.