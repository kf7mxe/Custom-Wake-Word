import wave
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def read(data_path):
    ''
    read
    voice
    signal
    '''
    wavepath = data_path
    f = wave.open(wavepath,'rb')
    params = f.getparams()
    Nchannels, sampwidth, framerate, nframes = params [: 4] ා number of channels, quantization bits, sampling frequency, sampling points
    STR? Data = f.readframes (nframes)? Read audio, string format
    f.close()
    Wavedata = np.fromstring (str_data, dtype = NP. Short) ා convert string to floating-point data
    Wavedata = wavedata * 1.0 / (max (ABS (wavedata))) × wave amplitude normalization
    return wavedata,nframes,framerate
   
   def enframe(data,win,inc):
    '' frame the voice data
    Input: data (one-dimensional array): voice signal
      WLAN (int): sliding window length
      Inc (int): the length of each window move
    Output: F (two-dimensional array) a two-dimensional array composed of data in each sliding window
    '''
    Nx = len(data) × length
    of
    voice
    signal
    try:
        nwin = len(win)
    except Exception as err:
        nwin = 1
    if nwin == 1:
        wlen = win
    else:
        wlen = nwin
    NF = int(NP.Fix((NX - WLAN) / Inc) + 1)
    times
    of
    window
    movement
    F = NP.Zeros((NF, WLAN)) ? initialize
    2
    D
    array
    indf = [inc * j for j in range(nf)]
    indf = (np.mat(indf)).T
    inds = np.mat(range(wlen))
    indf_tile = np.tile(indf, wlen)
    inds_tile = np.tile(inds, (nf, 1))
    mix_tile = indf_tile + inds_tile
    f = np.zeros((nf, wlen))
    for i in range(nf):
        for j in range(wlen):
            f[i, j] = data[mix_tile[i, j]]
    return f


def point_check(wavedata, win, inc):
    ''
    voice
    signal
    endpoint
    detection
    Input: wavedata(one - dimensional
    array): original
    voice
    signal
    Output: startpoint(int): start
    endpoint
    Endpoint(int): endpoint


'''
#1. Calculate the short-time zero crossing rate
FrameTemp1 = enframe(wavedata[0:-1],win,inc)
FrameTemp2 = enframe(wavedata[1:],win,inc)
Signs = NP. Sign (NP. Multiply (frametemp1, frametemp2)) ? calculate whether each bit of data adjacent to it has a different sign, and the different sign will cross zero
signs = list(map(lambda x:[[i,0] [i>0] for i in x],signs))
signs = list(map(lambda x:[[i,1] [i<0] for i in x], signs))
diffs = np.sign(abs(FrameTemp1 - FrameTemp2)-0.01)
diffs = list(map(lambda x:[[i,0] [i<0] for i in x], diffs))
zcr = list((np.multiply(signs, diffs)).sum(axis = 1))
#2. Calculate short-term energy
amp = list((abs(enframe(wavedata,win,inc))).sum(axis = 1))
#Set threshold
#Print ('set threshold ')
Zcrlow = max ([round (NP. Mean (ZCR) * 0.1), 3]) low threshold of zero crossing rate
Zcrhigh = max ([round (max (ZCR) * 0.1), 5]) high threshold of zero crossing rate
Amplow = min ([min (AMP) × 10, NP. Mean (AMP) × 0.2, max (AMP) × 0.1]) energy low threshold
Amphigh = max ([min (AMP) × 10, NP. Mean (AMP) × 0.2, max (AMP) × 0.1]) high energy threshold
#Endpoint detection
Maxsilence = 8 × maximum voice gap time
Minaudio = 16 ා minimum voice time
Status = 0 ා status 0: Mute segment, 1: transition segment, 2: voice segment, 3: end segment
Holdtime = 0 ා voice duration
Silence time = 0 ා voice gap time
Print ('Start endpoint detection ')
StartPoint = 0
for n in range(len(zcr)):
 if Status ==0 or Status == 1:
  if amp[n] > AmpHigh or zcr[n] > ZcrHigh:
   StartPoint = n - HoldTime
   Status = 2
   HoldTime = HoldTime + 1
   SilenceTime = 0
  elif amp[n] > AmpLow or zcr[n] > ZcrLow:
   Status = 1
   HoldTime = HoldTime + 1
  else:
   Status = 0
   HoldTime = 0
 elif Status == 2:
  if amp[n] > AmpLow or zcr[n] > ZcrLow:
   HoldTime = HoldTime + 1
  else:
   SilenceTime = SilenceTime + 1
   if SilenceTime < MaxSilence:
    HoldTime = HoldTime + 1
   elif (HoldTime - SilenceTime) < MinAudio:
    Status = 0
    HoldTime = 0
    SilenceTime = 0
   else:
    Status = 3
 elif Status == 3:
  break
 if Status == 3:
  break
HoldTime = HoldTime - SilenceTime
EndPoint = StartPoint + HoldTime
return FrameTemp1[StartPoint:EndPoint]


def mfcc(FrameK,framerate,win):
'' extract MFCC parameters 
Input: framek (two-dimensional array): two-dimensional framing speech signal
  Framerate: voice sampling frequency
  Win: framing window length (FFT points)
output:
'''
# Mel filter
mel_bank, w2 = mel_filter(24, win, framerate, 0, 0.5)
FrameK = FrameK.T
# Calculate power spectrum
S = abs(np.fft.fft(FrameK, axis=0)) ** 2
# Pass the power spectrum through the filter
P = np.dot(mel_bank, S[0:w2, :])
Take
a
logarithm
logP = np.log(P)
# Calculate DCT coefficient
# rDCT = 12
# cDCT = 24
# dctcoef = []
# for i in range(1,rDCT+1):
#  tmp = [np.cos((2*j+1)*i*math.pi*1.0/(2.0*cDCT)) for j in range(cDCT)]
#  dctcoef.append(tmp)
# Take a logarithm后做余弦变换
# D = np.dot(dctcoef,logP)
num_ceps = 12
D = dct(logP, type=2, axis=0, norm='ortho')[1:(num_ceps + 1), :]
return S, mel_bank, P, logP, D


def mel_filter(M, N, fs, l, h):
    ''
    Mel
    filter
    Input: m(int): number
    of
    filters
    N(int): FFT
    points
    FS(int): sampling
    frequency
    L(float): low
    frequency
    coefficient
    H(float): high
    frequency
    coefficient


Output: melbank(2
D
array): Mel
filter
'''
FL = FS * l ා lowest frequency in filter range
FH = FS * h ා highest frequency of filter range
BL = 1125 * np.log (1 + FL / 700)
bh = 1125 * np.log(1 + fh /700) 
B = BH - BL band width
Y = NP. Linspace (0, B, M + 2) ා mark Mel equally
Print ('mel interval ', y)
FB = 700 * (NP. Exp (Y / 1125) - 1) change Mel to Hz
print(Fb)
w2 = int(N / 2 + 1)
df = fs / N
Freq = [] ා sampling frequency value
for n in range(0,w2):
 freqs = int(n * df)
 freq.append(freqs)
melbank = np.zeros((M,w2))
print(freq)

for k in range(1,M+1):
 f1 = Fb[k - 1]
 f2 = Fb[k + 1]
 f0 = Fb[k]
 n1 = np.floor(f1/df)
 n2 = np.floor(f2/df)
 n0 = np.floor(f0/df)
 for i in range(1,w2):
  if i >= n1 and i <= n0:
   melbank[k-1,i] = (i-n1)/(n0-n1)
  if i >= n0 and i <= n2:
   melbank[k-1,i] = (n2-i)/(n2-n0)
 plt.plot(freq,melbank[k-1,:])
plt.show()
return melbank,w2

if __name__ == '__main__':
data_path = 'audio_data.wav'
win = 256
inc = 80
wavedata,nframes,framerate = read(data_path)
FrameK = point_check(wavedata,win,inc)
S,mel_bank,P,logP,D = mfcc(FrameK,framerate,win)