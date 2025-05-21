''' 
Module Speech/Audio Quality Parameters
'''

from __future__ import print_function
from __future__ import absolute_import 
from __future__ import division

import numpy as np
import subprocess, os
from scipy.signal import medfilt, correlate, correlation_lags, find_peaks
from scipy.io import wavfile
from scipy.stats import moment

from paralinguistic.ProsodyParameters import *
from utils.fe import windowing2
from utils.vad_ import compute_vad 
from utils.diariza import audio_from_maskdiar

import matplotlib.pyplot as plt

class Quality:
# Microcuts
    def mcuts_histogramenergy(self, x, fs, frame=0.016, shift=0.008, medfilt_order=3, min_edge=0, max_edge=20, th=0.01, mcuts_th = 0.13, val=-150):
        N=int(frame*fs)    #128
        X = windowing2(x, fs, frame, shift) * np.hamming(N) 
        E_target = 10*np.log10(np.sum(X*X,axis=1))
        E_target[E_target < val] = val
        E_target_medfiltered = medfilt(E_target,medfilt_order)
        
        #histograms
        theta_target = E_target_medfiltered-E_target
        alpha_target=theta_target[np.abs(theta_target)>th]
        x_axis=np.linspace(min_edge,max_edge,10)
        n_tar=np.histogram(alpha_target,x_axis)
        f_tar=n_tar[0]/np.max(n_tar[0])
        mcuts_metric = np.sum(f_tar[4:])

        if mcuts_metric > mcuts_th:
            mcuts = 1
        else:
            mcuts = 0
            
        return mcuts

# SNR estimation from snr definition using VAD y Diarization mask
    def snr_vad_diar(self, x, fs, mask, xspk1, xspk2):
        newnoise = x[(mask==0)]  # Localiza las zonas de silencio mezclando la mascara de diarizacion con el VAD
        N = np.mean(np.power(newnoise,2))
        
        SN1 = np.mean(np.power(xspk1,2)) 
        SNR1 = 10*np.log10((SN1/N)-1)
        SN2 = np.mean(np.power(xspk2,2))        
        SNR2 = 10*np.log10((SN2/N)-1)
        
        return np.round(SNR1,2), np.round(SNR2,2)
    
    def snr_vad_diar_stereo(self, x, fs, mask):
        N = np.mean(np.power(x[(mask==0)],2))
        SN = np.mean(np.power(x[(mask==1)],2)) 
        val = (SN/N)-1
        if val <= 0:
            SNR = -20
        else: 
            SNR = 10*np.log10(val)                
        return np.round(SNR,2)

# SNR estimation WADA algorithm ----------------------------------------------------------------------------------------------------------------------
    def wada_snr(self, x, fs=8000, vad=None):
        from Alpha04 import tabla_alpha_04_db as dbvals
        from Alpha04 import tabla_alpha_04_val as Gvals

        if vad is None:
            vad = np.asarray(np.ones(len(x)), dtype=np.int16)
        
        xx = x[vad==1]
        xx = np.asarray(xx, dtype=np.float32)
        abs_x = np.abs(xx)
        abs_x = [1e-10 if value <= 1e-10 else value for value in abs_x]

        dVal1 = np.mean(abs_x)
        dVal2 = np.mean(np.log(abs_x))
        dEng = np.sum(np.power(abs_x, 2))
        dVal3 = np.log(dVal1) - dVal2

        dSNRix = [index for index, val in enumerate(Gvals) if val < dVal3]
        if len(dSNRix) == 0:
            dSNR = dbvals[0]
        elif len(dSNRix) == len(dbvals):
            dSNR = dbvals[-1]
        else:
            dSNRix = np.max(dSNRix)
            dSNR = dbvals[dSNRix] \
                + (dVal3 - Gvals[dSNRix]) / (Gvals[dSNRix + 1] - Gvals[dSNRix]) \
                * (dbvals[dSNRix + 1] - dbvals[dSNRix])

        dFactor = 10**(dSNR / 10)
        dNoiseEng = dEng / (1 + dFactor)
        dSigEng = dEng * dFactor / (1 + dFactor)

        SNR = 10 * np.log10(dSigEng / dNoiseEng)
        return SNR

    def speechenergy(self, x, fs, frame=0.016, shift=0.008, val=-150, applyvad=True):
        if applyvad:
            vad, _ = compute_vad(x, fs)
        else:
            vad = np.asarray(np.ones(len(x)), dtype=np.int16)        
        x = x[vad==1]
        x = np.asarray(x, dtype=np.float32)
        
        N=int(frame*fs)    #128
        X = windowing2(x, fs, frame, shift) * np.hamming(N) 
        E = 10*np.log10(np.sum(X*X,axis=1))
        #E = 10*np.log10(np.mean(X*X,axis=1))
        E[E < val] = val
        return E
    
    def speechenergy_patent(self, x, fs, frame=0.016, shift=0.008, val=-150):
        N=int(frame*fs)    #128
        X = windowing2(x, fs, frame, shift) * np.hamming(N) 
        E = 10*np.log10(np.mean(X*X)/(2^29))
        if E < val: E = val
        return E
    
    def p563(self, x, fs, exec_dir='src/quality/p563/bin'):
        wav = 'tmp.wav'
        wavfile.write(wav, fs, x)
        command_line = [exec_dir+'/p563',wav]
        out = subprocess.run(command_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        p563_value = float(out.stdout.split('\t')[-2])
        os.remove(wav)
        return np.round(p563_value,2)
    
    def xcorr(self, x,y):
        """
        Perform Cross-Correlation on x and y
        x    : 1st signal
        y    : 2nd signal

        returns
        lags : lags of correlation
        corr : coefficients of correlation
        """
        corr = correlate(x, y, mode="full")
        lags = correlation_lags(len(x), len(y), mode="full")
        return lags, corr

    def crosstalk_stereo(self, x_ch0, x_ch1, fs, mask_ch0, mask_ch1, frame=1.5, shift=0.75, th_xcorrnorm=0.15, th_xcorrnorm_moment=3, debug=False):
        N=int(frame*fs)    #8000 (para 1 segundo de tiempo)
        X0 = windowing2(x_ch0, fs, frame, shift) 
        X1 = windowing2(x_ch1, fs, frame, shift) 
        
        Ntok = X0.shape[0]
        #Xcorr = np.zeros((Ntok,2*N))
        #Xlags = np.zeros((Ntok,2*N))
        #peaks = np.zeros(Ntok)
        tini = 0
        maxlag, maxcorr, maxcorr_norm, crosstalk_ch0, crosstalk_ch1, xcorrnorm_moment = [], [], [], [], [], []
        for i in range(Ntok):
            Xlags, Xcorr = Quality().xcorr(X0[i,:], X1[i,:])

            normindex = np.sqrt(np.sum(x_ch0**2)*np.sum(x_ch1**2))
            if normindex == 0: normindex = 0.0000001 # to avoid ceros
            Xcorr_norm = Xcorr/normindex

            peaks, _ = find_peaks(Xcorr,prominence=1) 
            #top_peaks_value = -np.sort(-Xcorr[peaks])[0:10]
            #top_peaks = np.argsort(-Xcorr[peaks])[0:10]       

            maxcorr.append(np.max(Xcorr)) # Valorar si esto es una crosstalk
            maxcorr_norm.append(np.max(Xcorr_norm))
            maxlag.append(tini + (np.argmax(Xcorr)/2-1)/fs) # Usar el lag para localizarlo (segundos)
            xcorrnorm_moment.append(moment(Xcorr_norm, moment=4))

            if debug: 
                xcorr_moment = moment(Xcorr, moment=4)
                if not os.path.exists('tmp_crosstalk'): os.makedirs('tmp_crosstalk')
                plt.figure(1)
                plt.subplot(121)
                plt.plot(Xlags, Xcorr)
                plt.title('Xcorr m4=' + str(round(xcorr_moment,2)))
                plt.subplot(122)
                plt.plot(Xlags, Xcorr_norm)
                plt.title('Norm Xcorr m4=' + str(round(xcorrnorm_moment[i],2)))
                plt.savefig('tmp_crosstalk/xcorr_'+str(i)+'.png')
                plt.clf()
                plt.figure(2)
                plt.plot(Xlags, Xcorr)
                plt.plot(peaks-len(Xlags)/2-1, Xcorr[peaks], "xr") 
                plt.title('Xcorr')
                plt.savefig('tmp_crosstalk/xcorr_peaks'+str(i)+'.png') 
                plt.clf()      

            # Channel: The crosstalk is in the channel with less speech (vad==0), but it disturb the speaker of the other channel
            # Value: There is crosstalk if the correlation is high and the 4th moment is high (so Xcorr distribution is considerably peaky) 
            if maxcorr_norm[i] > th_xcorrnorm and xcorr_moment[i]>th_xcorrnorm_moment:
                if sum(mask_ch0[tini:tini+N]) > sum(mask_ch1[tini:tini+N]):
                    crosstalk_ch0.append(0)
                    crosstalk_ch1.append(1)
                else:
                    crosstalk_ch0.append(1)
                    crosstalk_ch1.append(0)
            else:   
                crosstalk_ch0.append(0)
                crosstalk_ch1.append(0)

            tini += shift
        
        return crosstalk_ch0, crosstalk_ch1, maxlag, maxcorr, maxcorr_norm, xcorrnorm_moment



        




        
            
