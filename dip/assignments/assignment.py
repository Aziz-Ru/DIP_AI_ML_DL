import cv2 as cv
import numpy as np

def non_linear(img):
    norm = img.astype(np.float32) / 255.0
    c = 1.5
    log = c * np.log2(1 + norm)
    log_img = np.uint8(cv.normalize(log, None, 0, 255, cv.NORM_MINMAX))
    return log_img

def draw_combined_histogram(image, width=320, height=150):
    hist_img = np.zeros((height, 256, 3), dtype=np.uint8)

    for i, color in enumerate(('b', 'g', 'r')):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        hist = cv.normalize(hist, hist).flatten()

        for x in range(256):
            val = int(hist[x] * height)
            if color == 'b':
                cv.line(hist_img, (x, height), (x, height - val), (255, 0, 0))
            elif color == 'g':
                cv.line(hist_img, (x, height), (x, height - val), (0, 255, 0))
            elif color == 'r':
                cv.line(hist_img, (x, height), (x, height - val), (0, 0, 255))

    
    hist_img_resized = cv.resize(hist_img, (width, height))
    return hist_img_resized

def show_camera():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        linear_frame = cv.add(frame, 100)

        b, g, r = cv.split(frame)
        b_nl = non_linear(b)
        g_nl = non_linear(g)
        r_nl = non_linear(r)
        nonlinear_frame = cv.merge((b_nl, g_nl, r_nl))

        
        size = (320, 240)
        frame_resized = cv.resize(frame, size)
        linear_resized = cv.resize(linear_frame, size)
        nonlinear_resized = cv.resize(nonlinear_frame, size)

       
        hist_orig = draw_combined_histogram(frame_resized, width=320)
        hist_linear = draw_combined_histogram(linear_resized, width=320)
        hist_nl = draw_combined_histogram(nonlinear_resized, width=320)

        
        img_with_hist1 = np.vstack((frame_resized, hist_orig))
        img_with_hist2 = np.vstack((linear_resized, hist_linear))
        img_with_hist3 = np.vstack((nonlinear_resized, hist_nl))

        
        final_display = np.hstack((img_with_hist1, img_with_hist2, img_with_hist3))

        
        cv.imshow("Original | Linear | Non-Linear (With Histograms)", final_display)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    show_camera()
