import os
import cv2
import numpy as np
from tqdm import tqdm


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
    disp = kpNew - kp
    return np.expand_dims(np.median(disp, 0), 0)

def gaussianWeight(t, r, smoothingRadius):
    return np.exp((-9*(r-t)**2)/smoothingRadius**2)

def warpSingleFrame(frame, kps, disp):
    pt0 = np.asarray([
        [0, 0],
        [0, frame.shape[1] - 1],
        [frame.shape[0] - 1, 0],
        [frame.shape[0]-1, frame.shape[1] - 1]
    ])

    displacement = np.asarray([
        [disp[0], disp[1]]
    ])

    pt1 = pt0 + displacement

    M, mask = cv2.findHomography(pt0, pt1, cv2.RANSAC)
    warpedFrame = cv2.warpPerspective(frame , M, (frame.shape[1], frame.shape[0]))

    return warpedFrame

def warpFrameAlongSmoothPath(frames, kps, validFrames, smooth):
    stabilizedFrames = []

    print(len(frames), len(kps))
    for i in range(len(kps)):
        frame = frames[validFrames[i]]
        kp = kps[i]

        stabilizedFrames.append(warpSingleFrame(frames[i], kps[i], smooth[i]))

    return stabilizedFrames

def optimizePath(c, iterations=100, window_size=6):
    """
    @param: c is original camera trajectory
    @param: window_size is the hyper-parameter for the smoothness term

    This function was referenced from:
    https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization

    Returns:
            returns an optimized gaussian smooth camera trajectory
    """
    lambda_t = 100

    W = np.zeros((c.shape[0], c.shape[0]))
    for t in range(W.shape[0]):
        for r in range(-window_size//2, window_size//2+1):
            if t+r < 0 or t+r >= W.shape[1] or r == 0:
                continue
            W[t, t+r] = gaussianWeight(t, t+r, window_size)

    gamma = np.expand_dims(1 + lambda_t*np.dot(W, np.ones((c.shape[0],))), -1)

    p = c.copy()
    for iteration in range(iterations):
        p = np.divide(c+lambda_t*np.dot(W, p), gamma)

    return p

def optimizePath2(c, iterations=100, window_size=6):
    """
    @param: c is original camera trajectory
    @param: window_size is the hyper-parameter for the smoothness term


    Returns:
            returns an optimized bilateral smooth camera trajectory
    """
    lambda_t = 100
    t = c.shape[0]


    alpha = 0.0005
    W = np.asarray(
        [[r, r] for r in range(-window_size//2, window_size//2+1)]
    )
    W = np.exp((-9*(W)**2)/window_size**2)

    print(W.shape)
    meanKernal = np.ones((window_size + 1, 2)) / (window_size + 1)

    p = c.copy()
    gamma = np.expand_dims(1 + lambda_t* W.sum(0) , 0)

    for iteration in range(iterations):
        #print(cv2.filter2D(p, -1, meanKernal)[:10])

        var = p - cv2.filter2D(p, -1, meanKernal)
        weighted = cv2.filter2D(var, -1, W)

        bi = p - c

        p -= alpha*((p * bi) + (lambda_t * p * var))

    return p

def MotionBasedStabilization(originalFrames, motion=None):
    if len(originalFrames) == 0:
        return []

    if motion is None:
        motion = np.zeros((1, 2))

    kpList = []
    validFrames = []

    for i in range(0, len(originalFrames) - 1):
        kpOld, kpNew = LKOpticalFlow(originalFrames[i], originalFrames[i+1])

        if kpOld.shape[0] > 0:
            kpList.append(kpOld)
            validFrames.append(i)
            disp = getFeatureMotionVectors(kpOld, kpNew)
            motion = np.append(motion, motion[-1] + disp, 0)

    newMotion = motion[-len(originalFrames):]
    smoothMotion = optimizePath(newMotion)
    updateMotion = smoothMotion - newMotion

    stabilizedFrames = warpFrameAlongSmoothPath(originalFrames, kpList, validFrames, updateMotion)

    return stabilizedFrames

"================================== Experiment =============================="
def spatialWarpSingleFrame(frame, kps, displacement):
    pt0 = np.asarray([
        [0, 0],
        [0, frame.shape[1] - 1],
        [frame.shape[0] - 1, 0],
        [frame.shape[0]-1, frame.shape[1] - 1]
    ])

    update = np.asarray([
        [displacement[0], displacement[2]],
        [displacement[1], displacement[2]],
        [displacement[0], displacement[3]],
        [displacement[1], displacement[3]],
    ])

    pt1 = pt0 + update

    M, mask = cv2.findHomography(pt0, pt1, cv2.RANSAC)
    warpedFrame = cv2.warpPerspective(frame , M, (frame.shape[1], frame.shape[0]))

    return warpedFrame

def spatialWarpFrameAlongSmoothPath(frames, validFrames, kps, updateMotion):
    stabilizedFrames = []

    print(len(frames), len(kps))

    for i in range(len(kps)):
        frame = frames[validFrames[i]]
        kp = kps[i]

        stabilizedFrames.append(
            spatialWarpSingleFrame(frames[i], kps[i], updateMotion[i, :])
        )

    return stabilizedFrames

def getFeatureMotionVectorsSpatial(kpOld, kpNew, frameShape):
    disp = kpNew - kpOld

    left = disp[kpOld[:, 0] < frameShape[1]//2, 0]
    right = disp[kpOld[:, 0] >= frameShape[1]//2, 0]

    up = disp[kpOld[:, 1] < frameShape[0]//2, 1]
    down = disp[kpOld[:, 1] >= frameShape[0]//2, 1]

    leftMed = np.median(left) if left.shape[0] > 0 else np.median(disp[:, 0])
    rightMed = np.median(right) if right.shape[0] > 0 else np.median(disp[:, 0])

    upMed = np.median(up) if up.shape[0] > 0 else np.median(disp[:, 1])
    downMed = np.median(down) if down.shape[0] > 0 else np.median(disp[:, 1])

    return np.asarray([[leftMed, rightMed, upMed, downMed]])


def spatialMotionStabilization(originalFrames):
    if len(originalFrames) == 0:
        return []

    motion = np.zeros((1, 4))
    kpList = []
    validFrames = []

    for i in range(0, len(originalFrames) - 1):
        kpOld, kpNew = LKOpticalFlow(originalFrames[i], originalFrames[i+1])

        if kpOld.shape[0] > 0:
            kpList.append(kpOld)
            validFrames.append(i)
            motionVectors = getFeatureMotionVectorsSpatial(kpOld, kpNew, originalFrames.shape)
            motion = np.append(motion, motion[-1, :] + motionVectors, 0)

    smoothMotion = optimizePath(motion)
    updateMotion = smoothMotion - motion

    return spatialWarpFrameAlongSmoothPath(originalFrames, validFrames, kpList, updateMotion)
    #return originalFrames

"=============================================================================================="

def getVideoFrame(vpath, cap=None, maxFrames = 300):

  Framelist = []
  index = 0

  if cap is None:
      cap = cv2.VideoCapture(vpath)
      # Check if camera opened successfully
      if (cap.isOpened() == False):
        print("Error opening video stream or file")

  # Read until video is completed
  while(cap.isOpened() and index < maxFrames):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
      # Display the resulting frame
      Framelist.append(frame)

      index += 1
      # Press Q on keyboard to  exit
      if index >= maxFrames:
        return np.asarray(Framelist), cap

    # Break the loop
    else:
      break

  # When everything done, release the video capture object
  cap.release()

  # Closes all the frames
  cv2.destroyAllWindows()
  finalFramelist = np.array(Framelist)

  return finalFramelist, cap

def convertFramesToVideo(frames, path, fileName, out=None):
  if out is not None and len(frames) == 0:
      out.release()
      return
  elif out is None and len(frames) > 0:
      height, width, layers = frames[0].shape
      size = (width,height)

      fourecc = cv2.VideoWriter_fourcc(*'mp4v')
      out = cv2.VideoWriter(os.path.join(path, fileName), fourecc, 30, size)

  for i in range(len(frames)):
      out.write(frames[i])

  return out

def chunkStabilizeVideo(path, filename):
    vpath = os.path.join(path, filename)
    out = None
    cap = None
    frames = [0]
    motion = np.zeros((1, 2))

    while len(frames) > 0:
        print("frames processed:", len(frames))
        frames, cap = getVideoFrame(vpath, cap)
        #stabilizedFrames = spatialMotionStabilization(frames)
        stabilizedFrames = MotionBasedStabilization(frames, motion)
        out = convertFramesToVideo(stabilizedFrames, path, "out.mp4", out)









class Stabilizer:
    def __init__(self, path, inName, outName):
        self.inPath = os.path.join(path, inName)
        self.outPath = os.path.join(path, outName)

    def saveStabileVideo(self, validFrames, kps, updateMotion):
        cap = cv2.VideoCapture(self.inPath)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            return

        fourecc = cv2.VideoWriter_fourcc(*'mp4v')

        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)

        out = cv2.VideoWriter(self.outPath, fourecc, fps, size)

        ret, frame = cap.read()

        for i, valid in enumerate(tqdm(validFrames)):
            if valid:
                newFrame = warpSingleFrame(frame, kps[i], updateMotion[i])
                out.write(newFrame)

            ret, frame = cap.read()
            i += 1

        cv2.destroyAllWindows()
        cap.release()
        out.release()


    def estimatedMotionPath(self, kpList, validFrames):
        cap = cv2.VideoCapture(self.inPath)
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            return

        motion = np.zeros((1, 2))
        ret, frameOld = cap.read()
        ret, frameNew = cap.read()

        while ret:
            kpOld, kpNew = LKOpticalFlow(frameOld, frameNew)

            if kpOld.shape[0] > 0:
                kpList.append(kpOld)
                validFrames.append(True)
                motionVectors = getFeatureMotionVectors(kpOld, kpNew)
                motion = np.append(motion, motion[-1] + motionVectors, 0)
            else:
                validFrames.append(False)

            frameOld = frameNew
            ret, frameNew = cap.read()

        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()

        return motion


    def stabilizeVideo(self):
        kpList = []
        validFrames = []

        motion = self.estimatedMotionPath(kpList, validFrames)

        smoothMotion = optimizePath(motion)
        updateMotion = smoothMotion - motion

        self.saveStabileVideo(validFrames, kpList, updateMotion)


def stabilize(path, inName, outName):
    s = Stabilizer(path, inName, outName)
    s.stabilizeVideo()
