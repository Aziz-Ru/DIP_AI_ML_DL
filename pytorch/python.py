import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create figure
plt.ion()
fig, ax = plt.subplots()

# Canvas size
height, width = 200, 600

for x in range(0, width, 5):  # move step by step
    # Create black image
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Put text "Aziz"
    cv2.putText(
        img,
        "Aziz",
        (x, 100),  # moving x position
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),  # green color
        3,
        cv2.LINE_AA
    )

    # Convert BGR -> RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show image
    ax.clear()
    ax.imshow(img_rgb)
    ax.axis("off")

    plt.pause(0.03)

plt.ioff()
plt.show()