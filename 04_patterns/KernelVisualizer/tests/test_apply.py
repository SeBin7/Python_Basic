import numpy as np
from kernel_visualizer.apply import conv2d
from kernel_visualizer.kernels import get_kernel

def test_conv_gray_identity_same_shape():
    img = (np.random.rand(16, 16) * 255).astype(np.uint8)
    k = get_kernel("identity_3x3")
    out = conv2d(img, k, padding="same", keep_uint8=True)
    assert out.shape == img.shape

def test_conv_rgb_box_blur_uint8():
    img = (np.random.rand(10, 12, 3) * 255).astype(np.uint8)
    k = get_kernel("box_blur_3x3")
    out = conv2d(img, k, padding="same", keep_uint8=True)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
