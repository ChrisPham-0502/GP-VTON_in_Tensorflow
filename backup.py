# File này chứa các hàm chuyên biệt thay thế một số hàm PyTorch không được hỗ trợ trong Tensorflow

import tensorflow as tf
import tensorflow_addons as tfa

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros'):
    """
    Thực hiện lấy mẫu trên lưới từ đầu vào dữ liệu đa chiều.
    
    Args:
        input: Tensor, dữ liệu đầu vào có kích thước (B, H, W, C).
        grid: Tensor, tọa độ mục tiêu trên lưới, kích thước (B, H', W', 2).
        mode: str, chế độ nội suy, có thể là 'bilinear' hoặc 'nearest'.
        padding_mode: str, chế độ đối xử với các điểm nằm ngoài phạm vi ảnh, 
                      có thể là 'zeros' hoặc 'border'.
        
    Returns:
        Tensor, tensor kết quả có kích thước (B, H', W', C).
    """
    if mode == 'bilinear':
        method = tf.image.ResizeMethod.BILINEAR
    else:
        raise ValueError("Invalid mode. Mode must be 'bilinear' or 'nearest'.")

    if padding_mode == 'zeros':
        input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', constant_values=0)
    elif padding_mode == 'border':
        input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    else:
        raise ValueError("Invalid padding_mode. Padding_mode must be 'zeros' or 'border'.")

    # Scale grid to range [-1, 1]
    grid = tf.clip_by_value(grid, -1, 1)

    # Convert grid coordinates from [-1, 1] to [0, H-1] and [0, W-1]
    grid = (grid + 1) * tf.constant([input.shape[1] - 1, input.shape[2] - 1], dtype=tf.float32) / 2

    # Perform sampling
    '''
    - tensorflow 2.x không còn hỗ trợ contrib nên không thể dùng tf.contrib.resampler.
    Vậy nên có thể thay thế bằng tfa.image.resampler()'''
    sampled = tfa.image.resampler(input, grid)
    return sampled

class ModuleList(tf.Module):
    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is None:
            self.modules = []
        else:
            self.modules = modules

    def append(self, module):
        self.modules.append(module)

    def __getitem__(self, idx):
        return self.modules[idx]

    def __len__(self):
        return len(self.modules)

def interpolate(input, size=None, scale_factor=None, mode='bilinear'):
    """
    Thực hiện việc nội suy trên tensor đầu vào.

    Args:
        input: Tensor, dữ liệu đầu vào có kích thước (B, H, W, C).
        size: Tuple hoặc list chứa kích thước mới của ảnh (H', W').
        scale_factor: Tuple hoặc list chứa tỷ lệ co giãn cho mỗi chiều của ảnh.
        mode: str, chế độ nội suy, có thể là 'nearest', 'bilinear', hoặc 'bicubic'.

    Returns:
        Tensor, tensor kết quả có kích thước mới.
    """
    if size is None and scale_factor is None:
        raise ValueError("One of size or scale_factor must be provided.")
    
    if size is not None and scale_factor is not None:
        raise ValueError("Only one of size or scale_factor should be provided.")

    if size is not None:
        height, width = size
    else:
        height = tf.cast(tf.shape(input)[1] * scale_factor[0], dtype=tf.int32)
        width = tf.cast(tf.shape(input)[2] * scale_factor[1], dtype=tf.int32)

    if mode == 'nearest':
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif mode == 'bilinear':
        method = tf.image.ResizeMethod.BILINEAR
    else:
        raise ValueError("Invalid mode. Mode must be 'nearest', 'bilinear'.")

    resized = tf.image.resize(input, [height, width], method=method, antialias=True)

    return resized