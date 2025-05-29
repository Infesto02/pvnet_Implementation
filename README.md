**PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation**
# What is PVNet trying to do?

PVNet is a method to figure out:
"Where is an object in 3D space and how is it rotated?"

But it does that using **just a normal RGB image
This is called **6DoF pose estimation**, where:
- 3DoF = Position (X, Y, Z)
- 3DoF = Rotation (Yaw, Pitch, Roll)

# Why is this useful?

This is super important for things like:
- Robotics  (e.g., picking up or placing objects)
- Augmented reality (placing virtual objects on real surfaces)
 How does PVNet actually work?

1. The object in the image has some **known 3D keypoints** (e.g., corners, tips)
2. PVNet doesn't directly predict where those keypoints are
3. Instead, it makes **each pixel on the object predict a small arrow (direction)** pointing toward each keypoint
4. These arrows **vote together** to estimate the 2D position of the keypoints
5. Once we know where the 2D keypoints are (in the image), and we already know where those keypoints are in 3D (from the model), we use a method called **PnP** to estimate the full 6DoF pose

---

# Coordinate system confusion (that I finally understood!)

PVNet gives the pose of the object relative to the camera.

- That means the camera is considered to be at (0, 0, 0)
- We find out where the object is and how it is turned, as seen from the camera’s point of view

In contrast, some other methods (like PoseNet) predict the camera's pose relative to the object. That really confused me at first, but now I get it.

---

# 2D vs 3D keypoints

Another thing that helped me:
- 3D keypoints = points on the object (from the 3D model), like the tip of a tool
- 2D keypoints = where those points show up in the image

The whole idea is: if we can match 2D keypoints in the image to their known 3D positions, we can calculate the full pose.

#What does PVNet output?

At the end, PVNet gives:
- The object’s rotation (R)
- The object’s position (t)
- So together, you get the full 6DoF pose
- You can also get a mask showing which pixels belong to the object

---

# What's projection error?

When we estimate the pose, we want our predicted 2D keypoints (from projecting 3D ones using the pose) to be **close to where they actually are in the image**.

The distance between these is called projection error. Lower = better pose.

---

# The Paper

Here’s the original paper:  
PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation  
CVPR 2019 — [PDF Link](https://openaccess.thecvf.com/content_CVPR_2019/papers/Peng_PVNet_Pixel-Wise_Voting_Network_for_6DoF_Pose_Estimation_CVPR_2019_paper.pdf)


# pvnet_Implementation
