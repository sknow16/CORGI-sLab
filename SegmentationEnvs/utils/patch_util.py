# yuki作
import torch
import random

def random_coordinate(mask_point_range, batch):
    coordinates = [(random.randint(0, mask_point_range), random.randint(0, mask_point_range)) for _ in range(batch)]
    return coordinates

crop_color = 0.3
def create_crop(x, crop_size, mask=None):
    n, c, h, w = x.shape
    left_points = random_coordinate(w-crop_size, n)
    mini_x = torch.zeros(n, c, crop_size, crop_size)
    x_mask_ch = torch.zeros(n, c, h, w)
    if mask is not None:
        n, c, h, w = mask.shape
        mini_mask = torch.zeros(n, c, crop_size, crop_size)
    for i in range(n):
        mini_x[i] = x[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size]
        if mask is not None:
            mini_mask[i] = mask[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size]
        # x[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size] = 0
        x_mask_ch[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size] = torch.clamp(x_mask_ch[i, :, left_points[i][0]:left_points[i][0]+crop_size, left_points[i][1]:left_points[i][1]+crop_size]+1, 0, 1)
    new_x = torch.cat((x, x_mask_ch), dim=1)
    if mask is not None:
        return mini_x, new_x, mini_mask
    else:
        return mini_x, new_x , None

def inference_image_split(image, crop_size):
    coordinates = []
    _, _, height, width = image.shape
    for y in range(height-crop_size, -1, -crop_size):
        for x in range(width-crop_size, -1, -crop_size):
            coordinates.append((x, y))
    return coordinates

# オーバーラップの座標を出す
def overlap_image_split(image, crop_size):
    overlap_coordinates = []
    _, _, height, width = image.shape
    crop_size_half = crop_size/2
    crop_size_half = int(crop_size_half)
    
    pair0 = 0
    pair1 = crop_size_half
    while pair1 < height-crop_size_half:
        print(f"pair0: {pair0}, pair1: {pair1}")
        overlap_coordinates.append((pair0, pair1))
        overlap_coordinates.append((pair1, pair0))
        pair0 += crop_size_half
        pair1 += crop_size_half
    return overlap_coordinates

def create_inference_crop(x, crop_size, mask, left_point):
    n, c, h, w = x.shape
    mini_x = torch.zeros(n, c, crop_size, crop_size)
    x_mask_ch = torch.zeros(n, c, h, w)
    if mask is not None:
        n, c, h, w = mask.shape
        mini_mask = torch.zeros(n, c, crop_size, crop_size)
    mini_x    = x[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size]
    mini_mask = mask[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size]
    # x[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size] = torch.clamp(x[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size]+crop_color, 0, 1)
    x_mask_ch[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size] = torch.clamp(x_mask_ch[:, :, left_point[0]:left_point[0]+crop_size, left_point[1]:left_point[1]+crop_size]+1, 0, 1)
    new_x = torch.cat((x, x_mask_ch), dim=1)
    if mask is not None:
        return mini_x, new_x, mini_mask
    else:
        return mini_x, new_x, 

    