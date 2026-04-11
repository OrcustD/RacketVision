import os
import numpy as np


def list_dirs(directory):
    return sorted([os.path.join(directory, path) for path in os.listdir(directory)])


def to_img_format(input, num_ch, HEIGHT, WIDTH):
    """Transform model input sequence format to image sequence format.

    Args:
        input (numpy.ndarray): model input with shape ((N,) L*C, H, W)
        num_ch (int): Number of channels of each frame.

    Returns:
        (numpy.ndarray): Image sequences with shape ((N,) L, H, W) or ((N,) L, H, W, 3)
    """
    if num_ch == 1:
        return input
    else:
        add_axis = False
        if len(input.shape) == 3:
            add_axis = True
            input = input[np.newaxis, :, :]
        input = np.transpose(input, (0, 2, 3, 1))
        seq_len = int(input.shape[-1] / num_ch)
        img_seq = np.array([]).reshape(0, seq_len, HEIGHT, WIDTH, 3)
        for n in range(input.shape[0]):
            frame = np.array([]).reshape(0, HEIGHT, WIDTH, 3)
            for f in range(0, input.shape[-1], num_ch):
                img = input[n, :, :, f:f + 3]
                frame = np.concatenate((frame, img.reshape(1, HEIGHT, WIDTH, 3)), axis=0)
            img_seq = np.concatenate((img_seq, frame.reshape(1, seq_len, HEIGHT, WIDTH, 3)), axis=0)
        if add_axis:
            img_seq = img_seq.squeeze(0)
    return img_seq


def array_to_img(input):
    return (input * 255.0).astype(np.uint8)


def img_to_array(input):
    return input.astype(np.float32) / 255.0
