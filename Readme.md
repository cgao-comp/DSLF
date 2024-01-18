### DSLF Model Architecture Overview

The DSLF (Dual-channel Source Localization Framework) is designed for efficient source localization in multi-layer networks, particularly focusing on cross-platform data from Twitter and Weibo. The architecture comprises various neural network modules, each playing a crucial role in processing and analyzing propagation cascade data. We have conducted these experiments on the server with Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz and 188GB RAM. The integrated developing environment is PyTorch (the version of Torch is 1.12.0) on 3090Ti.
In our study, we employed two distinct models for graph-level and edge-level generation, detailed as follows:

#### Key Components of DSLF:

1. **LSTM_model1 and LSTM_model2:**
   - These are LSTM (Long Short-Term Memory) layers designed for sequence data processing.
   - Both models are configured with a `hidden_size` of 2 and 3 `num_layers`.
   - They are bidirectional, enhancing the capability to capture dependencies in both forward and backward directions.
   - Input features size is set to 15 for both models. Here, 15 includes 8 constructed dynamic propagation features (H<sub>1</sub>-H<sub>8</sub> in paper) and 7 user profiles (H<sub>9</sub>-H<sub>15</sub>), the verification status of a user is a binary variable, which takes 2 dimensions).
   - These models play a crucial role in understanding temporal dynamics in the data.

2. **Attention_GCN1 and Attention_GCN2 (Self-loop Attention-based GCN):**
   - Graph Convolutional Networks (GCNs) with a self-loop attention mechanism.
   - They consist of two layers of GCN with the number of input features (nfeat) set to 15 and the number of hidden units (nhid) calculated as `nfeat * ratio` (ratio is set to 2).
   - The dropout rate is set to 0.05 to prevent overfitting.
   - The attention mechanism is parameterized by `a`, a tensor of shape `[nhead, nfeat, 1]`, where `nhead` represents the number of attention heads.
   - These GCNs are crucial for capturing the structural dependencies in the network data.

3. **VAE1 and VAE2 (Variational Autoencoders):**
   - Designed to learn latent representations of the input data.
   - Both VAEs have an input size of 20 channels and a latent dimension size of 16.
   - The encoder and decoder in the VAEs consist of linear layers with ReLU activations.
   - These modules are vital for compressing high-dimensional data into a lower-dimensional latent space, facilitating efficient learning.

#### Hyperparameters and Dimensions:

- **LSTM Models:** 
  - Input Size: 15
  - Hidden Size: 2
  - Number of Layers: 3
  - Bidirectional: True

- **Attention-based GCN:**
  - Number of Heads for Attention: Configured as per requirement (nhead)
  - Input Features (nfeat): 15
  - Hidden Units (nhid): 30 (as nhid = nfeat * 2)
  - Dropout Rate: 0.05

- **VAEs:**
  - Input Channels: 20
  - Latent Dimension: 16
  - Encoder: Linear layers mapping from 20 to 64 and then to 32 (latent_dim * 2)
  - Decoder: Linear layers mapping from 16 to 64 and then to 20

#### Training and Testing Procedure:

- The framework undergoes a training phase followed by a testing phase, where it's evaluated on separate data loaders for Twitter and Weibo.
- A custom `collate_batch` function is used for preparing batch data.
- `Adam` optimizer is employed with a learning rate of 0.0005 for all model parameters.
- The models are trained for a predefined number of epochs 60 (`args.epochs`).

#### Output and Performance Metrics:

- The primary output includes the source localization probabilities.
- The performance is evaluated using the F-score metric, calculated based on the predicted and actual source nodes.

#### Code and Data Availability:

- The model is implemented in PyTorch.
- We have made our best effort to ensure the reproducibility of the code, subsequently ensuring a rigorous academic review and reproducibility of the manuscript.
Due to Twitter's policy restrictions and file size upload limitations, please contact the author for inquiries about the original dataset.

Fortunately, some datasets with existing links can be accessed through universally recognized ways:

[Twitter](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip)

[Weibo](https://www.dropbox.com/s/46r50ctrfa0ur1o/rumdect.zip?dl=0)
