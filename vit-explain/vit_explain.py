import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from timesformer.models.vit import TimeSformer
from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--mode', type=str, default='spatial_temporal', choices=['spatial_temporal', 'spatial_only'],
                        help='Which type of atttention to visualize')                    
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

import os

if __name__ == '__main__':
    args = get_args()

    # IMPORTANT: pre-trained model path
    model = TimeSformer(img_size=224, num_classes=5, num_frames=30, attention_type='divided_space_time',
      pretrained_model='/content/gdrive/MyDrive/Hand_Research/TimeSformer-main/output/checkpoints/No.3_acc=96%.pyth')
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # IMPORTANT: image dir path
    image_dir = "/content/gdrive/MyDrive/Hand_Research/TimeSformer-main/hand_data/val/WIN_20180907_15_30_06_Pro_Right_Swipe_new"
    
    # Get a list of all image files in the directory
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    image_tensor = []

    # Loop over the image files and stack them into the tensor
    for i, image_file in enumerate(image_files):
        with Image.open(image_file) as img:
            img = img.convert('RGB')  # Ensure all images have the same number of channels
            img = img.resize((224, 224))  # Resize image to the desired height and width
            img = torch.tensor(np.array(img).transpose(2, 0, 1)) 
            image_tensor.append(img)
    
    # Stack the image tensors along the third dimension
    stacked_tensor = torch.stack(image_tensor, dim=1)

    # Add an extra dimension to represent the batch size
    image_tensor = stacked_tensor.unsqueeze(0).float()
    if args.use_cuda:
      image_tensor = image_tensor.cuda()

    if args.category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, mode=args.mode,
            discard_ratio=args.discard_ratio)
        mask = attention_rollout(image_tensor)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio, mode=args.mode)
        mask = grad_rollout(image_tensor, args.category_index)
    
    # Loop over the image files and add the masks onto them
    for i, image_file in enumerate(image_files):
        with Image.open(image_file) as img:
            img = img.convert('RGB') 
            img = img.resize((224, 224))
            np_img = np.array(img)[:, :, ::-1]
            individual_mask = cv2.resize(mask[i], (np_img.shape[1], np_img.shape[0]))
            if args.category_index is None:
              name = "{:03d}_attention_rollout_{:.3f}_{}.png".format(i, args.discard_ratio, args.head_fusion)
            else: 
              name = "{:03d}_grad_rollout_{}_{:.3f}_{}.png".format(i, args.category_index, args.discard_ratio, args.head_fusion) 
            mask_img = show_mask_on_image(np_img, individual_mask)
            output_path = os.path.join('output/vit-explain', name)
            cv2.imwrite(output_path, mask_img)