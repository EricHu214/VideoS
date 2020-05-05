import os

import cv2
import numpy as np
import math
import ffmpeg


def LKOpticalFlow(frame1, frame2):
    frame = frame1.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # features that we can track between frames
    kp1 = cv2.goodFeaturesToTrack(
        frame,
        mask = None,
        maxCorners = 1000,
        qualityLevel = 0.3,
        minDistance = 7,
        blockSize = 7)

    nextFrame = frame2.copy()
    nextFrame = cv2.cvtColor(nextFrame, cv2.COLOR_BGR2GRAY)

    #using Lucas Kanade optical flow algorithm, find the same keypoints in the next frame. This can be done with
    # SIFT feature matching as well. Room for experimentation
    kp2, st, err = cv2.calcOpticalFlowPyrLK(
        frame,
        nextFrame,
        kp1,
        None,
        winSize  = (15, 15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    #print(kp1.shape, st.shape)
    return kp1[st==1], kp2[st==1]

# Using the keypoints in the old and new frame, get motion vectors
def getFeatureMotionVectors(kp, kpNew):
    x_displacements = []
    y_displacements = []

    kp_to_motion = {}

    for old, new in zip(kp, kpNew):
        x_displacements.append(new[0] - old[0])
        y_displacements.append(new[1] - old[1])

    return np.median(np.asarray(x_displacements)), np.median(np.asarray(y_displacements))

def gaussianWeight(t, r, smoothingRadius):
    return np.exp((-9*(r-t)**2)/smoothingRadius**2)

def warpSingleFrame(frame, kps, x_displacement, y_displacement):
    newkps = []

    pt0 = [
        [0, 0],
        [0, frame.shape[1] - 1],
        [frame.shape[0] - 1, 0],
        [frame.shape[0]-1, frame.shape[1] - 1]
    ]
    pt1 = [
        [0 + x_displacement, 0 + y_displacement],
        [0 + x_displacement, frame.shape[1] - 1 + y_displacement],
        [frame.shape[0] - 1 + x_displacement, 0 + y_displacement],
        [frame.shape[0]-1 + x_displacement, frame.shape[1] - 1 + y_displacement]
    ]

    M, mask = cv2.findHomography(np.asarray(pt0), np.asarray(pt1), cv2.RANSAC)
    warpedFrame = cv2.warpPerspective(frame , M, (frame.shape[1], frame.shape[0]))

    return warpedFrame

def warpFrameAlongSmoothPath(frames, kps, x_smooth, y_smooth):
    stabilizedFrames = []

    print(len(frames), len(kps))
    for i in range(len(frames) - 1):
        frame = frames[i]
        kp = kps[i]
        x_ = x_smooth[i]
        y_ = y_smooth[i]


        stabilizedFrames.append(warpSingleFrame(frames[i], kps[i], x_smooth[i], y_smooth[i]))

    return stabilizedFrames

def optimizeSinglePath(c, iterations=100, window_size=6):
    """
    @param: c is original camera trajectory
    @param: window_size is the hyper-parameter for the smoothness term


    Returns:
            returns an optimized gaussian smooth camera trajectory
    """
    lambda_t = 100
    p = np.empty_like(c)

    W = np.zeros((c.shape[0], c.shape[0]))
    for t in range(W.shape[0]):
        for r in range(-math.floor(window_size/2), math.floor(window_size/2)+1):
            if t+r < 0 or t+r >= W.shape[1] or r == 0:
                continue
            W[t, t+r] = gaussianWeight(t, t+r, window_size)

    gamma = 1+lambda_t*np.dot(W, np.ones((c.shape[0],)))

    #bar = tqdm(total=c.shape[0]*c.shape[1])
    Px = np.asarray(c[:, 0])
    Py = np.asarray(c[:, 1])
    for iteration in range(iterations):
        Px = np.divide(c[:, 0]+lambda_t*np.dot(W, Px), gamma)
        Py = np.divide(c[:, 1]+lambda_t*np.dot(W, Py), gamma)
    p[:, 0] = np.asarray(Px)
    p[:, 1] = np.asarray(Py)
    #bar.update(1)

    #bar.close()
    return p

def MotionBasedStabilization(originalFrames):
    x_motion = [0]
    y_motion = [0]

    kpList = []

    x_motion_stable = []
    y_motion_stable = []

    for i in range(0, len(originalFrames) - 1):
        kpOld, kpNew = LKOpticalFlow(originalFrames[i], originalFrames[i+1])

        kpList.append(kpOld)
        x_disp, y_disp = getFeatureMotionVectors(kpOld, kpNew)
        x_motion.append(x_motion[-1] + x_disp)
        y_motion.append(y_motion[-1] + y_disp)



    #x_smooth = GaussianMotion(np.asarray(x_motion))
    #y_smooth = GaussianMotion(np.asarray(y_motion))

    motion = np.empty((len(x_motion), 2))
    motion[:, 0] = x_motion
    motion[:, 1] = y_motion

    smoothMotion = optimizeSinglePath(motion)
    updateMotion = smoothMotion - motion

    stabilizedFrames = warpFrameAlongSmoothPath(originalFrames, kpList, updateMotion[:, 0], updateMotion[:, 1])

    ## for visualization
    #for i in range(len(stabilizedFrames) - 1):
    #    kpOld, kpNew = LKOpticalFlow(stabilizedFrames[i], stabilizedFrames[i+1])

    #    x_disp, y_disp = getFeatureMotionVectors(kpOld, kpNew)
    #    x_motion_stable.append(x_disp)
    #    y_motion_stable.append(y_disp)


    return stabilizedFrames


def getVideoFrame(vpath):

  Framelist = []
  cap = cv2.VideoCapture(vpath)
  # Check if camera opened successfully
  if (cap.isOpened()== False):
    print("Error opening video stream or file")

  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
      # Display the resulting frame
      Framelist.append(frame)
      # plt.imshow(frame,cmap='gray')
      # plt.show()

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Break the loop
    else:
      break

  # When everything done, release the video capture object
  cap.release()

  # Closes all the frames
  cv2.destroyAllWindows()
  finalFramelist = np.array(Framelist)

  return finalFramelist

def convertFramesToVideo(frames, path, fileName):
  height, width, layers = frames[0].shape
  size = (width,height)

  fourecc = 0x00000021
  #fourecc = cv2.VideoWriter_fourcc(*'h264')

  out = cv2.VideoWriter(os.path.join(path, fileName), fourecc, 30, size)

  for i in range(len(frames)):
      out.write(frames[i])

  out.release()

def stabilizeVideo(path, filename):
    vpath = os.path.join(path, filename)

    out, _ = (
        ffmpeg
        .input(vpath)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )
    video = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, height, width, 3])
    )

    print(video.shape)

    return

    frames = getVideoFrame(vpath)
    stabilizedFrames = MotionBasedStabilization(frames)
    convertFramesToVideo(stabilizedFrames, path, "out.mp4")
