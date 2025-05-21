from __future__ import division
import warnings
import numpy as np
from scipy import signal as sgn

# -------- Gammatone: filters ---------

DEFAULT_FILTER_NUM = 100
DEFAULT_LOW_FREQ = 100
DEFAULT_HIGH_FREQ = 44100 / 4

def erb_point(low_freq, high_freq, fraction):
    """
    Calculates a single point on an ERB scale between ``low_freq`` and
    ``high_freq``, determined by ``fraction``. When ``fraction`` is ``1``,
    ``low_freq`` will be returned. When ``fraction`` is ``0``, ``high_freq``
    will be returned.
    
    ``fraction`` can actually be outside the range ``[0, 1]``, which in general
    isn't very meaningful, but might be useful when ``fraction`` is rounded a
    little above or below ``[0, 1]`` (eg. for plot axis labels).
    """
    # Change the following three parameters if you wish to use a different ERB
    # scale. Must change in MakeERBCoeffs too.
    # TODO: Factor these parameters out
    ear_q = 9.26449 # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1

    # All of the following expressions are derived in Apple TR #35, "An
    # Efficient Implementation of the Patterson-Holdsworth Cochlear Filter
    # Bank." See pages 33-34.
    erb_point = (
        -ear_q * min_bw
        + np.exp(
            fraction * (
                -np.log(high_freq + ear_q * min_bw)
                + np.log(low_freq + ear_q * min_bw)
                )
        ) *
        (high_freq + ear_q * min_bw)
    )
    
    return erb_point

def erb_space(
    low_freq=DEFAULT_LOW_FREQ,
    high_freq=DEFAULT_HIGH_FREQ,
    num=DEFAULT_FILTER_NUM):
    """
    This function computes an array of ``num`` frequencies uniformly spaced
    between ``high_freq`` and ``low_freq`` on an ERB scale.
    
    For a definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983).
    "Suggested formulae for calculating auditory-filter bandwidths and
    excitation patterns," J. Acoust. Soc. Am. 74, 750-753.
    """
    return erb_point(
        low_freq,
        high_freq,
        np.arange(1, num + 1) / num
        )

def centre_freqs(fs, num_freqs, cutoff):
    """
    Calculates an array of centre frequencies (for :func:`make_erb_filters`)
    from a sampling frequency, lower cutoff frequency and the desired number of
    filters.
    
    :param fs: sampling rate
    :param num_freqs: number of centre frequencies to calculate
    :type num_freqs: int
    :param cutoff: lower cutoff frequency
    :return: same as :func:`erb_space`
    """
    return erb_space(cutoff, fs / 2, num_freqs)

def make_erb_filters(fs, centre_freqs, width=1.0):
    """
    This function computes the filter coefficients for a bank of 
    Gammatone filters. These filters were defined by Patterson and Holdworth for
    simulating the cochlea. 
    
    The result is returned as a :class:`ERBCoeffArray`. Each row of the
    filter arrays contains the coefficients for four second order filters. The
    transfer function for these four filters share the same denominator (poles)
    but have different numerators (zeros). All of these coefficients are
    assembled into one vector that the ERBFilterBank can take apart to implement
    the filter.
    
    The filter bank contains "numChannels" channels that extend from
    half the sampling rate (fs) to "lowFreq". Alternatively, if the numChannels
    input argument is a vector, then the values of this vector are taken to be
    the center frequency of each desired filter. (The lowFreq argument is
    ignored in this case.)
    
    Note this implementation fixes a problem in the original code by
    computing four separate second order filters. This avoids a big problem with
    round off errors in cases of very small cfs (100Hz) and large sample rates
    (44kHz). The problem is caused by roundoff error when a number of poles are
    combined, all very close to the unit circle. Small errors in the eigth order
    coefficient, are multiplied when the eigth root is taken to give the pole
    location. These small errors lead to poles outside the unit circle and
    instability. Thanks to Julius Smith for leading me to the proper
    explanation.
    
    Execute the following code to evaluate the frequency response of a 10
    channel filterbank::
    
        fcoefs = MakeERBFilters(16000,10,100);
        y = ERBFilterBank([1 zeros(1,511)], fcoefs);
        resp = 20*log10(abs(fft(y')));
        freqScale = (0:511)/512*16000;
        semilogx(freqScale(1:255),resp(1:255,:));
        axis([100 16000 -60 0])
        xlabel('Frequency (Hz)'); ylabel('Filter Response (dB)');
    
    | Rewritten by Malcolm Slaney@Interval.  June 11, 1998.
    | (c) 1998 Interval Research Corporation
    |
    | (c) 2012 Jason Heeris (Python implementation)
    """
    T = 1 / fs
    # Change the followFreqing three parameters if you wish to use a different
    # ERB scale. Must change in ERBSpace too.
    # TODO: factor these out
    ear_q = 9.26449 # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1

    erb = width*((centre_freqs / ear_q) ** order + min_bw ** order) ** ( 1 /order)
    B = 1.019 * 2 * np.pi * erb

    arg = 2 * centre_freqs * np.pi * T
    vec = np.exp(2j * arg)

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * np.cos(arg) / np.exp(B * T)
    B2 = np.exp(-2 * B * T)
    
    rt_pos = np.sqrt(3 + 2 ** 1.5)
    rt_neg = np.sqrt(3 - 2 ** 1.5)
    
    common = -T * np.exp(-(B * T))
    
    # TODO: This could be simplified to a matrix calculation involving the
    # constant first term and the alternating rt_pos/rt_neg and +/-1 second
    # terms
    k11 = np.cos(arg) + rt_pos * np.sin(arg)
    k12 = np.cos(arg) - rt_pos * np.sin(arg)
    k13 = np.cos(arg) + rt_neg * np.sin(arg)
    k14 = np.cos(arg) - rt_neg * np.sin(arg)

    A11 = common * k11
    A12 = common * k12
    A13 = common * k13
    A14 = common * k14

    gain_arg = np.exp(1j * arg - B * T)

    gain = np.abs(
            (vec - gain_arg * k11)
          * (vec - gain_arg * k12)
          * (vec - gain_arg * k13)
          * (vec - gain_arg * k14)
          * (  T * np.exp(B * T)
             / (-1 / np.exp(B * T) + 1 + vec * (1 - np.exp(B * T)))
            )**4
        )

    allfilts = np.ones_like(centre_freqs)
    
    fcoefs = np.column_stack([
        A0 * allfilts, A11, A12, A13, A14, A2*allfilts,
        B0 * allfilts, B1, B2,
        gain
    ])
    
    return fcoefs

def erb_filterbank(wave, coefs):
    """
    :param wave: input data (one dimensional sequence)
    :param coefs: gammatone filter coefficients
    
    Process an input waveform with a gammatone filter bank. This function takes
    a single sound vector, and returns an array of filter outputs, one channel
    per row.
    
    The fcoefs parameter, which completely specifies the Gammatone filterbank,
    should be designed with the :func:`make_erb_filters` function.
    
    | Malcolm Slaney @ Interval, June 11, 1998.
    | (c) 1998 Interval Research Corporation
    | Thanks to Alain de Cheveigne' for his suggestions and improvements.
    |
    | (c) 2013 Jason Heeris (Python implementation)
    """
    output = np.zeros((coefs[:,9].shape[0], wave.shape[0]))
    
    gain = coefs[:, 9]
    # A0, A11, A2
    As1 = coefs[:, (0, 1, 5)]
    # A0, A12, A2
    As2 = coefs[:, (0, 2, 5)]
    # A0, A13, A2
    As3 = coefs[:, (0, 3, 5)]
    # A0, A14, A2
    As4 = coefs[:, (0, 4, 5)]
    # B0, B1, B2
    Bs = coefs[:, 6:9]
    
    # Loop over channels
    for idx in range(0, coefs.shape[0]):
        # These seem to be reversed (in the sense of A/B order), but that's what
        # the original code did...
        # Replacing these with polynomial multiplications reduces both accuracy
        # and speed.
        y1 = sgn.lfilter(As1[idx], Bs[idx], wave)
        y2 = sgn.lfilter(As2[idx], Bs[idx], y1)
        y3 = sgn.lfilter(As3[idx], Bs[idx], y2)
        y4 = sgn.lfilter(As4[idx], Bs[idx], y3)
        output[idx, :] = y4 / gain[idx]
        
    return output


# -------- Gammatone: gtgram ---------

def round_half_away_from_zero(num):
    """ Implement the round-half-away-from-zero rule, where fractional parts of
    0.5 result in rounding up to the nearest positive integer for positive
    numbers, and down to the nearest negative number for negative integers.
    """
    return np.sign(num) * np.floor(np.abs(num) + 0.5)

def gtgram_strides(fs, window_time, hop_time, filterbank_cols):
    """
    Calculates the window size for a gammatonegram.
    
    @return a tuple of (window_size, hop_samples, output_columns)
    """
    nwin        = int(round_half_away_from_zero(window_time * fs))
    hop_samples = int(round_half_away_from_zero(hop_time * fs))
    columns     = (1
                    + int(
                        np.floor(
                            (filterbank_cols - nwin)
                            / hop_samples
                        )
                    )
                  )
        
    return (nwin, hop_samples, columns)


# -------- Gammatone: fftweight ---------

def specgram_window(
        nfft,
        nwin,
    ):
    """
    Window calculation used in specgram replacement function. Hann window of
    width `nwin` centred in an array of width `nfft`.
    """
    halflen = nwin // 2
    halff = nfft // 2 # midpoint of win
    acthalflen = int(np.floor(min(halff, halflen)))
    halfwin = 0.5 * ( 1 + np.cos(np.pi * np.arange(0, halflen+1)/halflen))
    win = np.zeros((nfft,))
    win[halff:halff+acthalflen] = halfwin[0:acthalflen];
    win[halff:halff-acthalflen:-1] = halfwin[0:acthalflen];
    return win

def specgram(x, n, sr, w, h):
    """ Substitute for Matlab's specgram, calculates a simple spectrogram.

    :param x: The signal to analyse
    :param n: The FFT length
    :param sr: The sampling rate
    :param w: The window length (see :func:`specgram_window`)
    :param h: The hop size (must be greater than zero)
    """
    # Based on Dan Ellis' myspecgram.m,v 1.1 2002/08/04
    assert h > 0, "Must have a hop size greater than 0"

    s = x.shape[0]
    win = specgram_window(n, w)

    c = 0

    # pre-allocate output array
    ncols = 1 + int(np.floor((s - n)/h))
    d = np.zeros(((1 + n // 2), ncols), np.dtype(complex))

    for b in range(0, s - n, h):
      u = win * x[b : b + n]
      t = np.fft.fft(u)
      d[:, c] = t[0 : (1 + n // 2)].T
      c = c + 1

    return d

def fft_weights(
    nfft,
    fs,
    nfilts,
    width,
    fmin,
    fmax,
    maxlen):
    """
    :param nfft: the source FFT size
    :param sr: sampling rate (Hz)
    :param nfilts: the number of output bands required (default 64)
    :param width: the constant width of each band in Bark (default 1)
    :param fmin: lower limit of frequencies (Hz)
    :param fmax: upper limit of frequencies (Hz)
    :param maxlen: number of bins to truncate the rows to
    
    :return: a tuple `weights`, `gain` with the calculated weight matrices and
             gain vectors
    
    Generate a matrix of weights to combine FFT bins into Gammatone bins.
    
    Note about `maxlen` parameter: While wts has nfft columns, the second half
    are all zero. Hence, aud spectrum is::
    
        fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft))
    
    `maxlen` truncates the rows to this many bins.
    
    | (c) 2004-2009 Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    | (c) 2012 Jason Heeris (Python implementation)
    """
    ucirc = np.exp(1j * 2 * np.pi * np.arange(0, nfft / 2 + 1) / nfft)[None, ...]
    
    # Common ERB filter code factored out
    cf_array = erb_space(fmin, fmax, nfilts)[::-1]

    _, A11, A12, A13, A14, _, _, _, B2, gain = (
        make_erb_filters(fs, cf_array, width).T
    )
    
    A11, A12, A13, A14 = A11[..., None], A12[..., None], A13[..., None], A14[..., None]

    r = np.sqrt(B2)
    theta = 2 * np.pi * cf_array / fs    
    pole = (r * np.exp(1j * theta))[..., None]
    
    GTord = 4
    
    weights = np.zeros((nfilts, nfft))

    weights[:, 0:ucirc.shape[1]] = (
          np.abs(ucirc + A11 * fs) * np.abs(ucirc + A12 * fs)
        * np.abs(ucirc + A13 * fs) * np.abs(ucirc + A14 * fs)
        * np.abs(fs * (pole - ucirc) * (pole.conj() - ucirc)) ** (-GTord)
        / gain[..., None]
    )

    weights = weights[:, 0:int(maxlen)]

    return weights, gain

def fft_gtgram(
    wave,
    fs,
    window_time, hop_time,
    channels,
    f_min):
    """
    Calculate a spectrogram-like time frequency magnitude array based on
    an FFT-based approximation to gammatone subband filters.

    A matrix of weightings is calculated (using :func:`gtgram.fft_weights`), and
    applied to the FFT of the input signal (``wave``, using sample rate ``fs``).
    The result is an approximation of full filtering using an ERB gammatone
    filterbank (as per :func:`gtgram.gtgram`).

    ``f_min`` determines the frequency cutoff for the corresponding gammatone
    filterbank. ``window_time`` and ``hop_time`` (both in seconds) are the size
    and overlap of the spectrogram columns.

    | 2009-02-23 Dan Ellis dpwe@ee.columbia.edu
    |
    | (c) 2013 Jason Heeris (Python implementation)
    """
    width = 1 # Was a parameter in the MATLAB code

    nfft = int(2 ** (np.ceil(np.log2(2 * window_time * fs))))
    nwin, nhop, _ = gtgram_strides(fs, window_time, hop_time, 0);

    gt_weights, _ = fft_weights(
            nfft,
            fs,
            channels,
            width,
            f_min,
            fs / 2,
            nfft / 2 + 1
        )

    sgram = specgram(wave, nfft, fs, nwin, nhop)

    result = gt_weights.dot(np.abs(sgram)) / nfft

    return result


# -------- Segment axis ---------

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.

    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is unevenly
    strided and being flattened or because end is set to 'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length:
        raise ValueError("frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0:
        raise ValueError("overlap must be nonnegative and length must "\
                          "be positive")

    if l < length or (l-length) % (length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length-overlap) \
               or (roundup == length and rounddown == 0)
        a = a.swapaxes(-1,axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s,dtype=a.dtype)
            if end in ['pad','wrap']:
                b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup-l]
            a = b
        elif end == 'delay':
            s = list(a.shape)
            l_orig = l
            l += overlap
            # if l not divisible by length, pad last frame with zeros
            if l_orig % (length-overlap):
                roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            else:
                roundup = l
            s[-1] = roundup
            b = np.empty(s,dtype=a.dtype)

            b[..., :(overlap)] = endvalue
            b[..., (overlap):(l_orig+overlap)] = a
            b[..., (l_orig+overlap):] = endvalue
            a = b
        else:
            raise ValueError("end has to be either 'cut', 'pad', 'wrap', or 'delay'.")

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l == 0:
        raise ValueError("Not enough data points to segment array in 'cut' mode; "\
              "try 'pad' or 'wrap'")
    assert l >= length
    assert (l-length) % (length-overlap) == 0
    n = 1 + (l-length) // (length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s,s) + a.strides[axis+1:]

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s,s) \
                     + a.strides[axis+1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)


# ------ Hilbert ---------

def hilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.
    The transformation is done along the last axis by default.
    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.
    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`
    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:
    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y
    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [1]_
    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``np.imag(hilbert(x))``, and the
    original signal from ``np.real(hilbert(x))``.
    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           http://en.wikipedia.org/wiki/Analytic_signal
    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
        # Make N multiple of 16 to make sure the transform will be fast
        if N % 16:
            N = int(np.ceil(N/16)*16)
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = np.fft(x, N, axis=axis)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if len(x.shape) > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    y = np.ifft(Xf * h, axis=axis)
    return y[:x.shape[axis]]


# ------ Modulation filters ---------

def make_modulation_filter(w0, Q):
    W0 = np.tan(w0/2)
    B0 = W0/Q
    b = np.array([B0, 0, -B0], dtype=np.float64)
    a = np.array([(1 + B0 + W0**2), (2*W0**2 - 2), (1 - B0 + W0**2)], dtype=np.float64)
    return b, a

def modulation_filterbank(mf, fs, Q):
    return [make_modulation_filter(w0, Q) for w0 in 2*np.pi*mf/fs]

def compute_modulation_cfs(min_cf, max_cf, n):
    spacing_factor = (max_cf/min_cf)**(1.0/(n-1))
    cfs = np.zeros(n)
    cfs[0] = min_cf
    for k in range(1,n):
        cfs[k] = cfs[k-1]*spacing_factor
    return cfs

def modfilt(F, x):
    y = np.zeros((len(F), len(x)), dtype=np.float64)
    for k, f in enumerate(F):
        y[k] = sgn.lfilter(f[0], f[1], x)
    return y


# ------ SRMR ---------

def calc_erbs(low_freq, fs, n_filters):
    ear_q = 9.26449 # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1

    erbs = ((centre_freqs(fs, n_filters, low_freq)/ear_q)**order + min_bw**order)**(1/order)
    return erbs

def calc_cutoffs(cfs, fs, q):
    # Calculates cutoff frequencies (3 dB) for 2nd order bandpass
    w0 = 2*np.pi*cfs/fs
    B0 = np.tan(w0/2)/q
    L = cfs - (B0 * fs / (2*np.pi))
    R = cfs + (B0 * fs / (2*np.pi))
    return L, R

def normalize_energy(energy, drange=30.0):
    peak_energy = np.max(np.mean(energy, axis=0))
    min_energy = peak_energy*10.0**(-drange/10.0)
    energy[energy < min_energy] = min_energy
    energy[energy > peak_energy] = peak_energy
    return energy

def srmr(x, fs, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=True, norm=False):
    wLengthS = .256
    wIncS = .064
    # Computing gammatone envelopes
    if fast:
        mfs = 400.0
        gt_env = fft_gtgram(x, fs, 0.010, 0.0025, n_cochlear_filters, low_freq)
    else:
        cfs = centre_freqs(fs, n_cochlear_filters, low_freq)
        fcoefs = make_erb_filters(fs, cfs)
        gt_env = np.abs(hilbert(erb_filterbank(x, fcoefs)))
        mfs = fs

    wLength = int(np.ceil(wLengthS*mfs))
    wInc = int(np.ceil(wIncS*mfs))

    # Computing modulation filterbank with Q = 2 and 8 channels
    mod_filter_cfs = compute_modulation_cfs(min_cf, max_cf, 8)
    MF = modulation_filterbank(mod_filter_cfs, mfs, 2)

    n_frames = int(1 + (gt_env.shape[1] - wLength)//wInc)
     
    w = sgn.windows.hamming(wLength+1)[:-1] # window is periodic, not symmetric

    energy = np.zeros((n_cochlear_filters, 8, n_frames))
    for i, ac_ch in enumerate(gt_env):
        mod_out = modfilt(MF, ac_ch)
        for j, mod_ch in enumerate(mod_out):
            mod_out_frame = segment_axis(mod_ch, wLength, overlap=wLength-wInc, end='pad')
            energy[i,j,:] = np.sum((w*mod_out_frame[:n_frames])**2, axis=1)

    if norm:
        energy = normalize_energy(energy)

    erbs = np.flipud(calc_erbs(low_freq, fs, n_cochlear_filters))

    avg_energy = np.mean(energy, axis=2)
    total_energy = np.sum(avg_energy)

    AC_energy = np.sum(avg_energy, axis=1)
    AC_perc = AC_energy*100/total_energy

    AC_perc_cumsum=np.cumsum(np.flipud(AC_perc))
    K90perc_idx = np.where(AC_perc_cumsum>90)[0][0]

    BW = erbs[K90perc_idx]

    cutoffs = calc_cutoffs(mod_filter_cfs, fs, 2)[0]

    if (BW > cutoffs[4]) and (BW < cutoffs[5]):
        Kstar=5
    elif (BW > cutoffs[5]) and (BW < cutoffs[6]):
        Kstar=6
    elif (BW > cutoffs[6]) and (BW < cutoffs[7]):
        Kstar=7
    elif (BW > cutoffs[7]):
        Kstar=8

    srmr_value = np.sum(avg_energy[:, :4])/np.sum(avg_energy[:, 4:Kstar])

    return srmr_value, energy




