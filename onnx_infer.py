import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from nets import ONNXSegmentModel

def mask_to_colormap(mask):
    # Convert label mask (1, H, W) to a colormap (H, W, 3)
    mask_clone = mask[0]    
    h, w = mask_clone.shape[0], mask_clone.shape[1]
    colormap = np.zeros((h, w, 3))
    colormap[mask_clone == 0] = [0, 0, 255]
    colormap[mask_clone == 1] = [0, 0, 0]
    return colormap.astype(np.uint8)

def mask_to_overlay_image(mask, image, background_id=1, ratio=0.55):
    # Convert label mask (1, H, W) to an overlay image
    colormap = mask_to_colormap(mask)
    overlay_image = image.copy()
    non_background_mask = (mask[0] != background_id)
    try:
        overlay_image[non_background_mask] = cv2.addWeighted(image[non_background_mask], ratio, colormap[non_background_mask], 1 - ratio, 0)
    except:
        pass
    return overlay_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple ONNX inference pipeline', fromfile_prefix_chars='@')
    parser.add_argument('-c', '--checkpoint_path', type=str, help='ONNX checkpoint path', default="yolov8n_lereslite.onnx")
    parser.add_argument('-n', help='if set, normalize the image', action='store_true')
    args = parser.parse_args()
    
    checkpoint_root = "weights"
    checkpoint_path = os.path.join(checkpoint_root, args.checkpoint_path)
    
    if os.path.isfile(checkpoint_path):
        model = ONNXSegmentModel(checkpoint_path=checkpoint_path, recover_original=True)
    else:
        print("== No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError
    print("== Loaded checkpoint '{}'".format(checkpoint_path))

    
    output_dir = "outs/1/"
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    input = "data_splits/idcard_test.txt"
    root = "../data/dataset"
    with open(input, "r") as f:
        samples = f.readlines()
    
    
    for sample in tqdm(samples):
        image_path, gt_path = sample.split()[:2]
        image = cv2.imread(f"{root}/{image_path}")
        mask_gt = cv2.imread(f"{root}/{gt_path}")
        
        basename = os.path.basename(image_path)

        mask_pred, h, w = model(image)

        mask_pred_vis = mask_to_colormap(mask_pred)
        overlay_ = mask_to_overlay_image(mask_pred, image)
        # disp_image_flat = np.hstack([image, mask_pred_vis, mask_gt, overlay_]) # all in one
        disp_image_flat = np.hstack([image, overlay_])
        disp_image_path = "{}/{}".format(output_dir, basename)
        cv2.imwrite(disp_image_path, disp_image_flat)
    