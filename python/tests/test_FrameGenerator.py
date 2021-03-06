import numpy as np
import loudness as ln

fs = 32000
N = 2048 + 32
x = np.arange(0, N)

# Input SignalBank
bufSize = 32
nSources = 1
nEars = 1
nChannels = 1
inputBank = ln.SignalBank()
inputBank.initialize(nSources, nEars, nChannels, bufSize, int(fs))

# Frame generator
frameSize = 2048
hopSize = 32
startAtWindowCentre = True
frameGen = ln.FrameGenerator(frameSize, hopSize, startAtWindowCentre)
frameGen.initialize(inputBank)
outputBank = frameGen.getOutput()

nBlocks = int(x.size / bufSize)
if startAtWindowCentre:
    nProcessedBlocks = int(nBlocks - 0.5 * frameSize / hopSize + 1)
else:
    nProcessedBlocks = int(nBlocks - frameSize / hopSize + 1)
frames = np.zeros((nEars, nProcessedBlocks, frameSize))

frameIndex = 0
for block in range(nBlocks):
    # Update input bank
    idx = block * bufSize
    inputBank.setSignal(0, 0, 0, x[idx:idx + bufSize])

    # process it
    frameGen.process(inputBank)

    # get output
    if(outputBank.getTrig()):
        sig = np.squeeze(outputBank.getSignals())
        frames[:, frameIndex, :] = np.copy(sig)
        frameIndex += 1

# Check frames are correct
if startAtWindowCentre:
    x = np.hstack((np.zeros(np.ceil((frameSize - 1) / 2.0)), x))

for ear in range(nEars):
    for i, frame in enumerate(frames[ear]):
        start = i * hopSize
        if all(frame != x[start:start + frameSize]):
            print("Frame number %d incorrect" % i)
