import glob
import cv2
import torch

# list every RGB frame in the TUM sequence and loads the first one as a BGR image
frames = sorted(glob.glob("data/rgbd_dataset_freiburg1_desk/rgb/*.png"))
img = cv2.imread(frames[0])

# Download the XFeat repo and pretrained weights from Github on the first run and model on the GPU
xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, top_k=1000).cuda()

# returns a list of one dictionary per image and takes the first one (keypoints, score, decriptors)
out = xfeat.detectAndCompute(img, top_k=1000)[0]

# Moves points back to the CPU as a Numpy Array and prints out how many that it found plus the shape
kps = out["keypoints"].cpu().numpy()
print(len(kps), "keypoints, descripter shape:", tuple(out["descriptors"].shape))

# draws a green dot at each of the keypoints and save the result
for x, y in kps:
    cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
cv2.imwrite("kps.png", img)
