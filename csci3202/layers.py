import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  row_dim = x.shape[0]
  col_dim = np.prod(x.shape[1:])
  x_reshape = x.reshape(row_dim, col_dim)
  out = np.dot(x_reshape, w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  x = cache[0]
  w = cache[1]
  b = cache[2]

  row_dim = x.shape[0]
  col_dim = np.prod(x.shape[1:])
  x_reshape = x.reshape(row_dim, col_dim)

  dw = x_reshape.T.dot(dout)
  dx = dout.dot(w.T).reshape(x.shape)
  db = np.sum(dout, axis=0)  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  x = cache
  out = np.maximum(0, x) # ReLU performed again
  out[out > 0 ] = 1
  dx = out * dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param["mode"]
  eps = bn_param.get("eps", 1e-5)
  momentum = bn_param.get("momentum", 0.9)

  N, D = x.shape
  running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == "train":
      # Compute output
      mu = x.mean(axis=0)
      xc = x - mu
      var = np.mean(xc ** 2, axis=0)
      std = np.sqrt(var + eps)
      xn = xc / std
      out = gamma * xn + beta

      cache = (mode, x, gamma, xc, std, xn, out)

      # Update running average of mean
      running_mean *= momentum
      running_mean += (1 - momentum) * mu

      # Update running average of variance
      running_var *= momentum
      running_var += (1 - momentum) * var
  elif mode == "test":
      # Using running mean and variance to normalize
      std = np.sqrt(running_var + eps)
      xn = (x - running_mean) / std
      out = gamma * xn + beta
      cache = (mode, x, xn, gamma, beta, std)
  else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param["running_mean"] = running_mean
  bn_param["running_var"] = running_var

  return out, cache

def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  mode = cache[0]
  if mode == "train":
      mode, x, gamma, xc, std, xn, out = cache

      N = x.shape[0]
      dbeta = dout.sum(axis=0)
      dgamma = np.sum(xn * dout, axis=0)
      dxn = gamma * dout
      dxc = dxn / std
      dstd = -np.sum((dxn * xc) / (std * std), axis=0)
      dvar = 0.5 * dstd / std
      dxc += (2.0 / N) * xc * dvar
      dmu = np.sum(dxc, axis=0)
      dx = dxc - dmu / N
  elif mode == "test":
      mode, x, xn, gamma, beta, std = cache
      dbeta = dout.sum(axis=0)
      dgamma = np.sum(xn * dout, axis=0)
      dxn = gamma * dout
      dx = dxn / std
  else:
      raise ValueError(mode)

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride

  out = np.zeros((int(N) , int(F) , int(H_out), int(W_out)))

  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
  for i in range(int(H_out)):
      for j in range(int(W_out)):
          x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
          for k in range(int(F)):
              out[:, k , i, j] = np.sum(x_pad_masked * w[k, :, :, :], axis=(1,2,3))
          
  out = out + (b)[None, :, None, None]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  
  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
  dx = np.zeros_like(x)
  dx_pad = np.zeros_like(x_pad)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  
  db = np.sum(dout, axis = (0,2,3))
  
  x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
  for i in range(int(H_out)):
      for j in range(int(W_out)):
          x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
          for k in range(int(F)): #compute dw
              dw[k ,: ,: ,:] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
          for n in range(int(N)): #compute dx_pad
              dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum((w[:, :, :, :] * 
                                                 (dout[n, :, i, j])[:,None ,None, None]), axis=0)
  dx = dx_pad[:,:,pad:-pad,pad:-pad]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  out_height = H / pool_height
  out_width = W / pool_width
  out = np.zeros((int(N), int(C), int(out_height), int(out_width)))
  for i in range(int(out_height)):
      for j in range(int(out_width)):
          mask = x[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
          out[:, :, i, j] = np.max(mask, axis=(2, 3))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  dx = np.zeros_like(x)
  out_height = H / pool_height
  out_width = W / pool_width
  for i in range(int(out_height)):
      for j in range(int(out_width)):
          # x, dx has the same dimension, so does x_mask and dx_mask
          x_mask = x[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
          dx_mask = dx[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
          # flags: only the max value is True, others are False
          flags = np.max(x_mask, axis=(2, 3), keepdims=True) == x_mask
          dx_mask += flags * (dout[:, :, i, j])[:, :, None, None]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  N, C, H, W = x.shape
  x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
  out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
  out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  return out, cache

def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  N, C, H, W = dout.shape
  dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
  dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
  dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
  return dx, dgamma, dbeta

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
