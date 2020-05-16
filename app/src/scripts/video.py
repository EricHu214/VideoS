import os
import cv2
import numpy as np
from tqdm import tqdm
import tempfile

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


def optimizePath(c, iterations=100, window_size=6, lambda_t=5):
    t = c.shape[0]
    alpha = 0.001

    # bilateral weights for the sum of differences
    W = np.asarray(
        [[r] for r in range(-window_size//2, window_size//2+1)]
    )
    W = np.exp((-9*(W)**2)/window_size**2)

    # sum of differences kernal for local patches of motion
    sumDiffKernal = -np.ones((window_size + 1, 1))
    sumDiffKernal[window_size//2 + 1] = window_size

    p = c.copy()

    # Iteratively minimize objective function proposed in the MeshFlow paper
    # using gradient descent
    for iteration in range(iterations):
        # gradient for local sum of square differences used as a smoothing factor
        # minimizes big jumps in motion between frames
        diff = cv2.filter2D(p, -1, sumDiffKernal)
        smooth = cv2.filter2D(diff, -1, W)

        # gradient for the anchor term to keep the optimized motion close to the
        # original camera path to reduce cropping
        anchor = p - c

        p -= alpha*((anchor) + (lambda_t * smooth))

    return p



class Stabilizer:
    def __init__(self, path, inName, outName, smoothness):
        self.inPath = os.path.join(path, inName)
        self.outPath = os.path.join(path, outName)
        self.smoothness = smoothness

    def cleanFiles(self):
        os.remove(self.inPath)
        os.remove(self.outPath)

    def generateStableVideo(self, validFrames, kps, updateMotion):
        cap = cv2.VideoCapture(self.inPath)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            return

        fourecc = cv2.VideoWriter_fourcc(*'mp4v')

        # get the info the video needed for the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)

        out = cv2.VideoWriter(self.outPath, fourecc, fps, size)

        ret, frame = cap.read()

        i = 0
        for valid in tqdm(validFrames):
            if valid:
                newFrame = warpSingleFrame(frame, kps[i], updateMotion[i])
                out.write(newFrame)
                i += 1

            ret, frame = cap.read()

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

        smoothMotion = optimizePath(motion, lambda_t=self.smoothness)
        updateMotion = smoothMotion - motion

        self.generateStableVideo(validFrames, kpList, updateMotion)


def stabilize(videoFile, smoothness):
    inName = next(tempfile._get_candidate_names())
    outName = next(tempfile._get_candidate_names()) + '.mp4'
    path = "/tmp/"

    videoFile.save(os.path.join(path, inName))
    s = Stabilizer(path, inName, outName, int(smoothness))
    s.stabilizeVideo()

    out = open(s.outPath, "rb")

    s.cleanFiles()

    return out
