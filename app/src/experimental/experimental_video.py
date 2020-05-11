def warpFrameAlongSmoothPath(frames, kps, validFrames, smooth):
    stabilizedFrames = []

    print(len(frames), len(kps))
    for i in range(len(kps)):
        frame = frames[validFrames[i]]
        kp = kps[i]

        stabilizedFrames.append(warpSingleFrame(frames[i], kps[i], smooth[i]))

    return stabilizedFrames

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
