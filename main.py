from src import gray_scale
import os
import math

URL = r"https://www.dropbox.com/s/qhhlqcica1nvtaw/opencv_bootcamp_assets_NB1.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_NB1.zip")
def main():
   gray_scale.build_grayScale()
   # print(int(math.log2(1+23)))
if __name__ == "__main__":
    main()
