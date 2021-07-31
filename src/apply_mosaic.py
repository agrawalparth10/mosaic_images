import argparse
import matplotlib.pyplot as plt
from mosaic import mosaic
import os
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--source_dir",
    type=str, required=True, help='Path to directory of source images')
parser.add_argument("--target_dir",
    type=str, required=True, help='Path to directory of target images')
parser.add_argument("--alpha",
    type=float, default=0.5, help="Coefficient for linear combination [0-1]")
parser.add_argument("--grid_size",
    nargs=2, type=int, default=None, help='List of height and width of the target images')
parser.add_argument("--num_clusters",
    type=int, default=3, help='Number of clusters for k mediods')
parser.add_argument("--output_dir",
    type=str, default=None, help='Output directory to save the image')

args = parser.parse_args()

if __name__ == "__main__":

    # Mosaic class
    mosaic_model = mosaic(args.source_dir,
        args.target_dir,
        alpha=args.alpha,
        grid_size=args.grid_size,
        num_clusters=args.num_clusters)

    # Create the mosaic
    mosaic_imgs = mosaic_model.apply_mosaic()

    # Output directory
    output_dir = args.output_dir

    # Create the directory if doesn't exist
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Display and save the mosaics
    count = 0
    for mosaic_img in mosaic_imgs:
        plt.imshow(mosaic_img)
        plt.show()

        if output_dir is not None:
            im = Image.fromarray((mosaic_img * 255).astype(np.uint8))
            im.save(output_dir + '/mosaic_' + str(count) + '.png')

    print("Done!")