# Open CV

## cv2.imread() method - Python OpenCV

If the image cannot be read because of missing file, improper permissions or an unsupported/invalid format then it returns an empty matrix.

```
cv2.imread(filename, flag)
```

Parameters:

- filename: specifies the path to the image file.
  flag: specifies the way how the image should be read which can be :

- cv2.IMREAD_COLOR - It specifies to load a color image. Any transparency of image will be neglected. It is the default flag. Alternatively we can pass integer value 1 for this flag.

- cv2.IMREAD_GRAYSCALE - It specifies to load an image in grayscale mode. Alternatively we can pass integer value 0 for this flag.

- cv2.IMREAD_UNCHANGED - It specifies to load an image such as including alpha channel. Alternatively we can pass integer value -1 for this flag.

x -> image width
y -> image height

```
[
  [ [B,G,R], [B,G,R], [B,G,R] ],  # row 0
  [ [B,G,R], [B,G,R], [B,G,R] ],  # row 1
  [ [B,G,R], [B,G,R], [B,G,R] ],  # row 2
  [ [B,G,R], [B,G,R], [B,G,R] ]   # row 3
]

```

## img.shape

```
img.shape → (height, width, channels)
```

- height → number of rows (pixels vertically)

- width → number of columns (pixels horizontally)

- channels → number of color channels (usually 3 for BGR)
