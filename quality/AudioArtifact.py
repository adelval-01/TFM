'''
Metrics related to artifacts in the audio signal
'''
import pandas as pd
import numpy as np
from audiolazy.lazy_lpc import lpc
from srmr import srmr
from quality.Satdet import *
from quality.Quality import *
from paralinguistic.ProsodyParameters import *
from utils.Info import *
#from utils.vad_ import segment_vad, compute_vad
from utils.fe import preemphasis

class AudioArtifact:
    def compute_artifact(self, x, fs, cfg, mask):
        # Apply sppech/silence mask to signal
        
        tokensize=cfg["tokensize"] #tokensize = 2  # Analysis time for dump an artifact vector = 2 seconds
        nartifacts= 13 #cfg["nartifacts"] #Nartifact = 10  # Number of artifacts: tini, tout, dtmf, tones, noise, aloud, sat, microcuts
        tokensizemcuts = cfg["tokensizemcuts"] #tokensizemcuts = 10sec

        dur = len(x)/fs
        Ntoken = int(np.ceil(dur/tokensize))  

        if Ntoken > 1: 
            artifact = np.zeros((Ntoken,nartifacts))            
            # Parameters each tokensize seconds 
            ini = 0        
            Ntokenmcuts = int(np.ceil(dur/tokensizemcuts)) 
            output = []
            for i in range(0,Ntokenmcuts):
                out = ini + tokensizemcuts*fs
                xtok = x[ini:out]
                output.append(Quality().mcuts_histogramenergy(xtok, fs, frame=cfg["frame"], shift=cfg["shift"], 
                                                            medfilt_order=cfg["medfilt_order"], min_edge=cfg["min_edge"], max_edge=cfg["max_edge"], 
                                                            th=cfg["th"], mcuts_th=cfg["mcuts_th"], val=cfg["val"]))
                ini = out
            mcuts=np.kron(output,np.ones(int(tokensizemcuts/tokensize)))
            
            ini = 0
            df_artifact = pd.DataFrame(columns=['tini', 'tout', 'vad', 'noise', 'srmr', 'noisefloor', 'loudness', 'speechlevel', 'sat', 'tones', 'dtmf', 'mcuts', 'p563'])
            for i in range(0,Ntoken):
                out = ini + tokensize*fs
                xtok = x[ini:out]
                tini = ini/fs
                tout = out/fs
                # Check if the token is speech or silence
                masktok = mask[ini:out]
                if sum(masktok)/len(masktok) > 0.9: vad = 1
                elif sum(masktok)/len(masktok) < 0.1: vad = 0
                else: vad = 2 #mezcla de speech y silence
                
                dtmf, tones, noise, loudness, speechlevel, sat, srmr, noisefloor, p563 = AudioArtifact().artifact(xtok,fs)
                df_artifact.loc[i] = [tini, tout, vad, noise, srmr, noisefloor, loudness, speechlevel, sat, tones, dtmf, mcuts[i], p563]
                #artifact[i,:] = [tini, tout, vad, noise, srmr, loudness, noisefloor, sat, tones, dtmf, mcuts[i], p563]            
                ini = out

            # Summary for the full signal
            snr = Quality().wada_snr(x, fs, vad=mask) 
            silence_mask = mask.copy()
            silence_mask[mask==1] = 0
            silence_mask[mask==0] = 1
            snr_silence = Quality().wada_snr(x, fs, vad=silence_mask)
            p563 = 1.0 #Quality().p563(x, fs) es muy lento este algoritmo
            df_summary = Info().summary(x,fs,mask,df_artifact,snr,snr_silence,p563,tokensize)
        else:
            df_artifact = None
            df_summary = None

        return df_artifact, df_summary
    
    def artifact(self,x,fs):
        # DTMF, Tones, Puffs
        dtmf, tones = AudioArtifact().tones(x,fs)
        # Noise detection (with VAD)
        noise = AudioArtifact().noise(x,fs)
        # Noisefloor segmentation (Energy in silence segments)
        noisefloor_value = AudioArtifact().noisefloor(x, fs)
        # SRMR detection
        srmr_value = AudioArtifact().srmr(x,fs)
        # SpeakAloud detection (with VAD): Detect speech segment pretty aloud
        loudness = AudioArtifact().speakaloud(x,fs)
        # SpeakAloud detection (with VAD): Detect speech segment pretty aloud
        loudness_patent = AudioArtifact().speechlevel(x,fs)
        # Saturation detection 
        sat = AudioArtifact().saturation(x,fs)
        # p563 quality metric
        #try:
        #    p563 = Quality().p563(x, fs)
        #except:
        p563 = 1.0
        
        return dtmf, tones, noise, loudness, loudness_patent, sat, srmr_value, noisefloor_value, p563

    def tones(self, x, fs):        
        # LPC detect spiky peaks
        lpcorder = 2
        thmod = 0.9 # threshold for the root module, dtmf should be spiky, so high module
        thbw = 50 # threshold for bandwith, dtmf should be narrow 
        dtmf_freq = [697, 770, 852, 941, 1209, 1336, 1477, 1633]

        x_copy = x.copy()
        x_copy = preemphasis(x_copy)  # Enfatiza los tonos que estan en la alta frecuencia, para evitar que se fije mucho en el pitch o armonicos de baja frequencia propios de la voz
        tone = np.zeros(lpcorder)    
        dtmf = np.zeros(lpcorder)    
        A = lpc(x_copy,lpcorder)

        rts = A.numpolyz.roots
        angz = np.arctan2(np.imag(rts),np.real(rts))
        bw = -1/2*(fs/(2*np.pi))*np.log(np.abs(rts))
        mod = np.abs(A.numpolyz.roots)
        freq = np.sort(angz*(fs/(2*np.pi)))
        #Q = bw/freq

        for i in range(len(rts)):
            if mod[i]>thmod and bw[i]<thbw:
                tone[i] = 1
                if dtmf[i]==0:
                    for j in range(i+1,len(rts)):
                        if mod[i]-mod[j]<0.01 and bw[i]-bw[j]<1: # Asegura que sea una raiz compleja conjugada
                            # Chequea si el tono es un dtmf comparando con las freq dtmf estandares +- 10Hz 
                            for f in dtmf_freq:
                                if f-10 < freq[i] < f+10:
                                    dtmf[i] = 1
                                    dtmf[j] = 1
                                    tone[i] = 0
                            break
                else:
                    tone[i]=0
        flag1_dtmf = np.max(dtmf)
        flag1_tone = np.max(tone)
        flag2_tone, flag2_dtmf = 0, 0
        
        # Decision
        flag_tone = flag1_tone + flag2_tone
        flag_dtmf = flag1_dtmf + flag2_dtmf

        return flag_dtmf, flag_tone
    
    def saturation(self, x, fs, N=0.050, M=0.010):
        # Saturation detect
        L = len(x)
        shift = int(M*fs)
        window = int(N*fs)
        try:    
            det = SatDet_Info(fs, window, shift)
            sat = do_satdet(det, x, L)
            sat_index = np.sum(sat)/float(len(sat))
        except:
            sat_index = 0
        
        #if len(sat) > 0: return np.max(sat)
        #else: return 0
        
        # Correccion: Retorna segmento saturado cuando encontro minimo 5% de las ventanas de 50ms saturadas   
        if sat_index > 0.05: 
            return 1 
        else: 
            return 0
      
    def noise(self, x, fs):
        noise = Quality().wada_snr(x, fs, vad=None)
        return np.round(noise,2)
    
    def speakaloud(self, x, fs):
        # Intensity: Loudness        
        I = ProsodyParameters().intensity_praat(x,fs)
        intensity = np.median(I) 

        return np.round(intensity,2)   

    def noisefloor(self, x, fs, frame=0.016, shift=0.008, val=-150):
        # Compute noise floor
        N=int(frame*fs) #128
        X = windowing2(x, fs, frame, shift) * np.hamming(N) 
        E = 10*np.log10(np.sum(X*X,axis=1))
        E[E < val] = val # To avoid E=-inf
        E = np.median(E)
        return np.round(E,2)
        
    def srmr(self, x, fs):
        try:
            srmr_value, _ = srmr(x, fs, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=True, norm=False)
            srmr_value = np.round(srmr_value,2)
        except:
            srmr_value = np.nan
        return np.round(srmr_value,2)
    
    def speechlevel(self, x, fs):
        # Speech level of the signal as in the patent description: Should be similar than loudness but with maximum 0, as in audacity audio level        
        speechlevel = Quality().speechenergy_patent(x, fs)
        
        return np.round(speechlevel,2)  