from Layers import Base
import numpy as np
from scipy import signal
from copy import deepcopy


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.num_channels = self.convolution_shape[0]
        self.weights = np.random.uniform(low=0.0, high=1.0, size=(self.num_kernels, *self.convolution_shape))
        self.bias = np.random.uniform(low=0.0, high=1.0, size=(num_kernels,))
        self._gradient_weights = np.zeros(self.weights.shape)
        self._gradient_bias = np.zeros(self.bias.shape)
        self._optimizer = None
        self._bias_optimizer = None

        self.input_tensor = None
        self.img_shape = None
        self.forward_pad_sizes = None
        self.backward_pad_sizes = self.calculate_backward_pad_sizes(self.convolution_shape)
        self.is1D = True if len(self.convolution_shape) == 2 else False

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._bias_optimizer = deepcopy(optimizer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.img_shape = input_tensor[0, 0].shape
        if self.forward_pad_sizes == None:
            self.forward_pad_sizes = self.calculate_forward_pad_sizes(self.img_shape)


        # stride_shape_tuple = self.stride_shape
        # if isinstance(self.stride_shape, int):
        #     stride_shape_tuple = (self.stride_shape,)
        # output_shape = []
        # strides = []
        # for idx, img_ax_size in enumerate(self.img_shape):
        #     ax_output_shape = img_ax_size // stride_shape_tuple[idx]
        #     if img_ax_size % self.stride_shape[idx] != 0:
        #         ax_output_shape += 1
        #     output_shape.append(ax_output_shape)
        #     strides.append(input_tensor[0, 0].strides[idx] * stride_shape_tuple[idx])
        #
        # output_images = []
        # # for image in range(input_tensor.shape[0]):
        # for image in input_tensor:
        #     output_kernels = []
        #     for kernel in range(self.num_kernels):
        #         if isinstance(self.bias, float):
        #             merged_channels_output = self.bias
        #         else:
        #             merged_channels_output = self.bias[kernel]
        #         for channel in range(self.convolution_shape[0]):
        #             convoled_output = signal.correlate(image[channel], self.weights[kernel, channel],
        #                                                mode='same')
        #             merged_channels_output += np.lib.stride_tricks.as_strided(convoled_output, shape=output_shape,
        #                                                                       strides=strides)
        #
        #         # output_kernels.append(convoled_output)
        #         output_kernels.append(merged_channels_output)
        #     output_images.append(output_kernels)
        #
        # return np.asarray(output_images)


        padded_input = np.pad(input_tensor, self.forward_pad_sizes, constant_values=0)

        output_batch = None
        if not self.is1D:
            for kernel in range(self.num_kernels):
                merged_channels_output = self.bias[kernel]
                for channel in range(self.num_channels):
                    sliding_window_view = np.lib.stride_tricks.sliding_window_view(padded_input[:, channel],
                                                                                   self.convolution_shape[1:],
                                                                                   axis=(1, 2))[:,::self.stride_shape[0], ::self.stride_shape[1]]
                    convoled_output = np.zeros(sliding_window_view.shape[:-2])
                    for i in range(sliding_window_view.shape[0]):
                        for h in range(sliding_window_view.shape[1]):
                            for w in range(sliding_window_view.shape[2]):
                                convoled_output[i][h][w] = signal.correlate(sliding_window_view[i][h][w],
                                                                     self.weights[kernel, channel], mode='valid')

                    merged_channels_output += convoled_output
                if output_batch is  None:
                    output_batch = np.expand_dims(merged_channels_output,axis=1)
                else:
                    output_batch = np.concatenate((output_batch, np.expand_dims(merged_channels_output,axis=1)), axis=1)
        else:
            for kernel in range(self.num_kernels):
                merged_channels_output = self.bias[kernel]
                for channel in range(self.num_channels):
                    sliding_window_view = np.lib.stride_tricks.sliding_window_view(padded_input[:, channel],
                                                                                   self.convolution_shape[1:],
                                                                                   axis=(1,))[:, ::self.stride_shape[0]]
                    convoled_output = np.zeros(sliding_window_view.shape[:-1])
                    for i in range(sliding_window_view.shape[0]):
                        for h in range(sliding_window_view.shape[1]):
                            convoled_output[i][h] = signal.correlate(sliding_window_view[i][h], self.weights[kernel, channel], mode='valid')
                    merged_channels_output += convoled_output

                if output_batch is None:
                    output_batch = np.expand_dims(merged_channels_output, axis=1)
                else:
                    output_batch = np.concatenate((output_batch, np.expand_dims(merged_channels_output,axis=1)), axis=1)
        return np.asarray(output_batch)

    def backward(self, error_tensor):
        self.gradient_weights = np.zeros_like(self.weights)

        # gradient wrt weights
        for idx in range(error_tensor.shape[0]):
            input_img = self.input_tensor[idx]
            input_err = error_tensor[idx]
            for kernel_idx, err_channel in enumerate(input_err):
                for channel_idx, img_channel in enumerate(input_img):
                    img_channel = np.pad(img_channel, self.backward_pad_sizes, constant_values=0)
                    upsampled_error = self.upsample(err_channel, self.stride_shape)
                    self.gradient_weights[kernel_idx, channel_idx] += signal.correlate(img_channel, upsampled_error,
                                                                                       mode='valid')

        # gradient wrt bias
        if not self.is1D:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
        else:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

        # weights/bias update
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        # gradient wrt input
        # adjusting weights
        s_weights = np.split(self.weights, self.convolution_shape[0], axis=1)
        s_x_h_weights = np.stack(s_weights, axis=0)
        s_x_h_weights = np.squeeze(s_x_h_weights, axis=2)

        output_errors = []
        for err in error_tensor:
            output_kernels = []
            for kernel in range(self.num_channels):
                correlate_output = 0
                for channel in range(self.num_kernels):
                    upsampled_error = self.upsample(err[channel], self.stride_shape)
                    correlate_output += signal.convolve(upsampled_error, s_x_h_weights[kernel, channel], mode='same')
                output_kernels.append(correlate_output)
            output_errors.append(output_kernels)

        gradient_wrt_input = np.asarray(output_errors)

        return gradient_wrt_input

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.num_kernels, *self.convolution_shape),
                                                      np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize((self.num_kernels,))

    def upsample(self, error, stride):
        if stride == 1 or stride == (1, 1):
            return error
        upsampled_error = np.zeros(self.img_shape)
        if len(error.shape) == 2:
            err_h, err_w = error.shape
            for h in range(err_h):
                for w in range(err_w):
                    upsampled_error[h * stride[0], w * stride[1]] = error[h, w]
        if len(error.shape) == 1:
            err_h = error.shape[0]
            for h in range(err_h):
                upsampled_error[h * stride[0]] = error[h]
        return upsampled_error

    def calculate_backward_pad_sizes(self, convolution_shape):
        # 2d convolution
        if len(convolution_shape) == 3:
            _, convolution_h, convolution_w = convolution_shape
            # even kernel -> asymmetric padding
            if convolution_h % 2 == 0:
                h_pad_sizes = (convolution_h // 2, convolution_h // 2 - 1)
            else:
                h_pad_sizes = (convolution_h // 2, convolution_h // 2)
            if convolution_w % 2 == 0:
                w_pad_sizes = (convolution_w // 2, convolution_w // 2 - 1)
            else:
                w_pad_sizes = (convolution_w // 2, convolution_w // 2)

            return (h_pad_sizes, w_pad_sizes)
            # img_channel = np.pad(img_channel, (h_pad_sizes, w_pad_sizes), constant_values=0)
        # 1d convolution
        elif len(convolution_shape) == 2:
            _, convolution_h = convolution_shape
            if convolution_h % 2 == 0:
                pad_sizes = (convolution_h // 2, convolution_h // 2 - 1)
            else:
                pad_sizes = (convolution_h // 2, convolution_h // 2)
            return (pad_sizes,)


    def calculate_forward_pad_sizes(self, img_shape):
        pad_sizes = [(0,0),(0,0)]
        for i, img_axis_size in enumerate(img_shape):
            ax_output_shape = img_axis_size // self.stride_shape[i]
            if img_axis_size % self.stride_shape[i] != 0:
                ax_output_shape += 1
            pad_size_before = self.convolution_shape[i + 1] // 2
            pad_size_after =  (ax_output_shape -1) * self.stride_shape[i] + ((self.convolution_shape[i + 1] - 1)  // 2)  - self.img_shape[i] + 1
            pad_sizes.append((pad_size_before, pad_size_after))
        return tuple(pad_sizes)