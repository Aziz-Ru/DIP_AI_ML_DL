import numpy as np

def filter(input_img, kernel):
  tmp_img = input_img.astype(np.float32)
  input_h, input_w = input_img.shape
  kernel_h, kernel_w = kernel.shape
  output_h = input_h - kernel_h + 1
  output_w = input_w - kernel_w + 1

  output_img = np.zeros((output_h, output_w), dtype = np.float32)
  for h in range(output_h):
    for w in range(output_w):
      roi = tmp_img[h : h + kernel_h, w : w + kernel_w]
      output_img[h, w] = int(np.sum(roi * kernel))

  output_img = np.clip(output_img, 0, 255).astype(np.uint8)

  return output_img


