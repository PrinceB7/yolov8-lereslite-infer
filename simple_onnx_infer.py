import os
import cv2
import argparse
import numpy as np

from nets import ONNXSegmentModel

def mask_to_overlay_image(mask, image, background_id=1, ratio=0.50):
    # Convert label mask (1, H, W) to an overlay image
    mask_clone = mask[0]    
    h, w = mask_clone.shape[0], mask_clone.shape[1]
    colormap = np.zeros((h, w, 3))
    colormap[mask_clone == 0] = [0, 0, 255]
    colormap[mask_clone == 1] = [0, 0, 0]
    
    overlay_image = image.copy()
    non_background_mask = (mask[0] != background_id)
    try:
        overlay_image[non_background_mask] = cv2.addWeighted(image[non_background_mask], ratio, colormap.astype(np.uint8)[non_background_mask], 1 - ratio, 0)
    except:
        pass
    return overlay_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple ONNX inference pipeline', fromfile_prefix_chars='@')
    parser.add_argument('-c', '--checkpoint_path', type=str, help='ONNX checkpoint path', default="weights/yolov8n_lereslite.onnx")
    parser.add_argument('-i', '--image_path', type=str, help='input image path', default="data/5.png")
    args = parser.parse_args()
    
    if os.path.isfile(args.checkpoint_path):
        model = ONNXSegmentModel(checkpoint_path=args.checkpoint_path, recover_original=True)
    else:
        print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        raise FileNotFoundError
    print("== Loaded checkpoint '{}'".format(args.checkpoint_path))

    image = cv2.imread(args.image_path)
    mask_pred, h, w = model(image)
        
    overlay_image = mask_to_overlay_image(mask_pred, image)
    cv2.imshow("prediction", overlay_image)
    cv2.waitKey(0)
    