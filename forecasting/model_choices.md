Project Model Architecture Choices
==================================

Although RNNs work well with short-term sequences and theoretically would work pretty well on this task, LSTM layers work better because we are working with *multiple* time series, and it is beneficial for the network to remember previously trained-on data since there are similar trends across the data.

I chose to include a 1D CNN layer as a form of pre-processing on the sequences to pick out patterns, however it didn't seem to add much improvement to the model; if anything it was very slight. Adding more filters (kernels) than what already is set causes the model to lose accuracy since we are training on such short sequences of data that many kernels doesn't pick out aspects well. A smaller amount of kernels seems to add a little bit more value (slightly lower loss, and slightly more accurate-looking predictions to the observer).

A pure dense-layer network performs poorly and does not remember trends too well, since there are far too few parameters. LSTMs add many parameters which likely overfit themselves onto the dataset. 