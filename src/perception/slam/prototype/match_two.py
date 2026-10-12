import random
import glob
import cv2
import torch

# lists the TUM frames in a timestamp order as before and loads two of them
frames = sorted(glob.glob("data/rgbd_dataset_freiburg1_desk/rgb/*.png"))
im1, im2 = cv2.imread(frames[0]), cv2.imread(frames[30])

# loads XFeat model onto the actual GPU
xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, top_k=1000).cuda()

# XFWeat runs on each image and holes the image keypoints and scores
d1 = xfeat.detectAndCompute(im1, top_k=1000)[0]
d2 = xfeat.detectAndCompute(im2, top_k=1000)[0]

d1["image_size"] = (im1.shape[1], im1.shape[0])
d2["image_size"] = (im2.shape[1], im2.shape[0])

# LighterGlue compares two sets of keypoints and descriptors and returns the same matched points
res = xfeat.match_lighterglue(d1, d2)
print(len(res), "return values")
p1, p2 = res[0], res[1]
print(len(p1), "matches")

# sticks 2 images side by side together
canvas = cv2.hconcat([im1, im2])
w = im1.shape[1]

# for each matched pair draw a green line from the point in the left image to the partner in the right image
for (x1, y1), (x2, y2) in random.sample(list(zip(p1, p2)), min(100, len(p1))):
    cv2.line(canvas, (int(x1), int(y1)), (int(x2) + w, int(y2)), (0, 255, 0), 1)
cv2.imwrite("matches.png", canvas)
