# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
import scipy as sp
import scipy.signal as sig
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.mlab as m
import statsmodels.api as sm

## Constants
Fs = 200 # sampling frequency in Hz
F_nyq = Fs/2 # Nyquist frequency
Ts = 1./Fs # sampling period in seconds
EPOCH_START = -0.2 # in seconds relative to event
EPOCH_END = 0.8 # in seconds relative to event
BASELINE_START = -0.2 # in seconds relative to event
BASELINE_END = 0 # in seconds relative to event

Neg_ErrP_START = 0.325 # in seconds relative to feedback
Neg_ErrP_END = 0.435 # in seconds relative to feedback
Pos_ErrP_START = 0.550 # in seconds relative to feedback
Pos_ErrP_END = 0.650 # in seconds relative to feedback


# Filter options
N_ord = 4 # order of the filter
F_lo = 0.1 # cuttof frequency in Hz
F_hi = 30.0 # cuttof frequency in Hz

# Design the bandpass filter
(bpfilt_b, bpfilt_a) = sig.butter(N_ord, [F_lo/F_nyq, F_hi/F_nyq], btype='bandpass')


def compute_trial_idxs(sessions, feedbacknos):
    '''
    Given the numbers of sessions and trials in the sessions, gives absolute ordinal numbers of the trials.
    '''
    trial_idxs = []
    for (session, feedbackno) in zip(sessions, feedbacknos):
        if session == 5:
            trial_idxs.append(4 * 60 + feedbackno)
        else:
            trial_idxs.append((session - 1) * 60 + feedbackno)
    
    return trial_idxs
    
def lat2idx(lats, Fs):
    '''
    Latencies lats in seconds transformed to sample indices using sampling frequency Fs in Hz.
    '''
    return np.array([int(lat * Fs) for lat in lats])
    
    
def idx2lat(idxs, Fs):
    '''
    Sample indices idxs transformed to latencies in seconds using sampling frequency Fs in Hz.
    '''
    return np.array([float(idx) / Fs for idx in idxs])
    
    
def get_epochs(x, idxs, epoch_start, epoch_end):
    '''
    The indices and boundaries are given in samples.
    '''
    epoched_x = x[:, np.array([idx + np.arange(epoch_start, epoch_end + 1) for idx in idxs])]
    epoched_x = epoched_x.transpose([0, 2, 1]) # rearranges dimensions into: channel X time X epoch
    
    return epoched_x
    
    
def remove_baseline(epoched_x, epoch_start, baseline_start, baseline_end):
    '''
    Removes the mean value of the baseline period from the trial. All the times are given in number of samples.
    '''
    new_epoched_x = np.zeros_like(epoched_x)
    idx_start =  baseline_start - epoch_start
    idx_end = baseline_end - epoch_start
    for i_trial in range(np.size(epoched_x, 2)):
        for i_ch in range(np.size(epoched_x, 0)):
            baseline = np.mean(epoched_x[i_ch, idx_start:idx_end, i_trial])
            new_epoched_x[i_ch, :, i_trial] = epoched_x[i_ch, :, i_trial] - baseline
            
    return new_epoched_x
    
    
def mean_amplitude(epoched_x, epoch_start, peak_start, peak_end):
    '''
    Calculates mean amplitude for each channel and each trial of epoched x. Peak start and end are given in sample indices.
    '''
    idx_start =  peak_start - epoch_start
    idx_end = peak_end - epoch_start
    
    n_trial = np.size(epoched_x, 2)
    n_chan = np.size(epoched_x, 0)
    amps = np.zeros((n_trial, n_chan))
    for i_trial in range(n_trial):
        for i_chan in range(n_chan):
            amps[i_trial, i_chan] = np.mean(epoched_x[i_chan, idx_start:idx_end, i_trial])
            
    return amps
    


if __name__ == '__main__':
    # Load trial descriptions
    TrainLabels = pd.read_csv('../data/TrainLabels.csv', header=0)
    
    # Add columns, for subject, session, feedback number, type of trial, Cz ERP, Neg-ErrP and Pos-ErrP amplitudes
    TrainLabels['Subject'] = TrainLabels['IdFeedBack'].map(lambda s : int(s[1:3]))
    TrainLabels['Session'] = TrainLabels['IdFeedBack'].map(lambda s : int(s[8:10]))
    TrainLabels['FeedbackNo'] = TrainLabels['IdFeedBack'].map(lambda s : int(s[13:]))
    TrainLabels['AbsTrialNum'] = compute_trial_idxs(TrainLabels['Session'], TrainLabels['FeedbackNo'])
    
    #TrainLabels['Cz ERP'] = np.nan
    #TrainLabels['Cz ERP'] = TrainLabels['Cz ERP'].astype(object) # change dtype to object so it can contain np arrays
    TrainLabels['Neg-ErrP'] = np.nan
    TrainLabels['Pos-ErrP'] = np.nan
    
    TrainLabels = TrainLabels[['IdFeedBack', 'Subject','Session','FeedbackNo','AbsTrialNum','Prediction','Neg-ErrP','Pos-ErrP']]
    
    # Define grouping over subjects and sessions to access datasets
    TrainGroups = TrainLabels.groupby(['Subject','Session'])
    
    # Go through all datasets and compute Cz ERP, Neg-ErrP and Pos-ErrP
    Cz_ERP = dict()
    test_dict = {(2,2) : range(60, 121)}
    for ((subject, session), rows) in TrainGroups.groups.iteritems():
    #for ((subject, session), rows) in test_dict.iteritems():
        # Load the dataset
        dataset_path = os.path.join('..', 'data', 'train', 'Data_S%02d_Sess%02d.csv') % (subject, session)
        print "Processing: ", dataset_path
        
        SessionData = pd.read_csv(dataset_path, header=0, usecols=['Time', 'Cz', 'EOG', 'FeedBackEvent'])
        
        # Define variables of interest
        t_session = SessionData['Time']
        eog = SessionData['EOG']
        cz = SessionData['Cz']
        event_ch = SessionData['FeedBackEvent']
        event_times = SessionData.Time[SessionData.FeedBackEvent == 1] # in seconds
        event_idxs = lat2idx(event_times, Fs)  # in samples
        (epoch_start, epoch_end) = lat2idx((EPOCH_START, EPOCH_END), Fs)
        (baseline_start, baseline_end) = lat2idx((BASELINE_START, BASELINE_END), Fs)
        (negerrp_start, negerrp_end) = lat2idx((Neg_ErrP_START, Neg_ErrP_END), Fs)
        (poserrp_start, poserrp_end) = lat2idx((Pos_ErrP_START, Pos_ErrP_END), Fs)
        
        # Filter Cz and EOG
        eog_filt = sig.filtfilt(bpfilt_b, bpfilt_a, eog, padtype='even')
        cz_filt = sig.filtfilt(bpfilt_b, bpfilt_a, cz, padtype='even')
        
        # Regress EOG out of Cz
        mod = sm.OLS(cz_filt, eog_filt)   # Describe model (Cz = beta * Eog + eps)
        res = mod.fit()       # Fit model
        cz_clean = cz_filt - res.params[0] * eog_filt # Remove beta * EOG from Cz 

        
        # Split Cz into epochs
        cz_epochs = get_epochs(cz_clean[np.newaxis], event_idxs, epoch_start, epoch_end)
        
        # Remove baseline
        cz_epochs_nobase = remove_baseline(cz_epochs, epoch_start, baseline_start, baseline_end)
        
        # Calculate Neg-ErrP and Pos-ErrP mean amplitude
        negerrp_amps = mean_amplitude(cz_epochs_nobase, epoch_start, negerrp_start, negerrp_end)
        poserrp_amps = mean_amplitude(cz_epochs_nobase, epoch_start, poserrp_start, poserrp_end)
        
        # Save results into table
        for row  in rows:
            trial = TrainLabels.loc[row, 'FeedbackNo'] - 1;
            #TrainLabels.at[row, 'Cz ERP'] = cz_epochs_nobase[:, :, trial]
            Cz_ERP[row] = cz_epochs_nobase[:, :, trial]
            TrainLabels.at[row, 'Neg-ErrP'] = negerrp_amps[trial]
            TrainLabels.at[row, 'Pos-ErrP'] = poserrp_amps[trial]
            
    
    # Save the extended trial descriptions and signals
    results_path = os.path.join('..', 'results', 'trial_description.csv')
    TrainLabels.to_csv(results_path)
    
    cz_results_path = os.path.join('..', 'results', 'Cz_ERP.out')
    f = open(cz_results_path, 'w')
    pickle.dump(Cz_ERP, f)
    f.close()
        
    


    