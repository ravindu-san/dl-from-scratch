from Layers import Base
import numpy as np


class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.h_stride, self.w_stride = stride_shape
        self.pooling_shape = pooling_shape
        self.max_val_idxes = None
        self.input_tensor_shape = None

    def forward(self, input_tensor):
        self.input_tensor_shape = input_tensor.shape
        sliding_window_view = np.lib.stride_tricks.sliding_window_view(input_tensor, self.pooling_shape, axis=(2, 3))[:,
                              :, ::self.h_stride, ::self.w_stride, :, :]
        merged_window_view = sliding_window_view.reshape(*sliding_window_view.shape[:-2], np.prod(self.pooling_shape))
        self.max_val_idxes = np.argmax(merged_window_view, axis=4)
        return np.max(merged_window_view, axis=4)

    def backward(self, error_tensor):
        sliding_window_view = np.zeros((*error_tensor.shape, np.prod(self.pooling_shape)))
        np.put_along_axis(sliding_window_view, np.expand_dims(self.max_val_idxes, axis=-1),
                          np.expand_dims(error_tensor, axis=-1), axis=-1)
        sliding_window_view = sliding_window_view.reshape(*sliding_window_view.shape[:-1], *self.pooling_shape)
        upsampled_error = self.upsample(sliding_window_view)
        return upsampled_error

    def upsample(self, sliding_window_view):
        upsample_errs = np.zeros(self.input_tensor_shape)
        n_imgs, n_channels, height, width, _, _ = sliding_window_view.shape
        h_pool, w_pool = self.pooling_shape
        for i in range(n_imgs):
            for c in range(n_channels):
                for h in range(height):
                    for w in range(width):
                        h_up_start, h_up_end = h * self.h_stride, h * self.h_stride + h_pool
                        w_up_start, w_up_end = w * self.w_stride, w * self.w_stride + w_pool
                        upsample_errs[i, c, h_up_start:h_up_end, w_up_start:w_up_end] += sliding_window_view[i, c, h, w]

        return upsample_errs
