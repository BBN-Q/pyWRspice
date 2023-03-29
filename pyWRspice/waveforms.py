# Copyright (c) 2023 Raytheon BBN Technologies - Quantum Group

"""
    Generate waveforms as inputs to pyWRspice circuits
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, linspace, arange, floor, ceil, sin, cos
from pyWRspice.simulation import RawFile, values_to_binary, values_to_text, write_rawfile, Variable

class waveform_mod(object):
    # Waveform with a base frequency, where you apply modulations to mag and phase piecewise
    def __init__(self,freq, step, simtime, start_amp = 0.0, start_phase = 0.0, start_offset = 0.0, name=None):
        self.freq = freq
        self.step = step
        self.simtime = simtime
        self.name = name

        self.time = 1e-9*arange(0,simtime, step) # Give these in ns
        self.phase = np.empty(self.time.shape)
        self.phase[:] = start_phase
        self.amp = np.empty(self.time.shape)
        self.amp[:] = start_amp
        self.offset = np.empty(self.time.shape)
        self.offset[:] = start_offset

    def roll(self, ts, ph):
        i0 = floor(ts[0]/self.step).astype(int)
        i1 = floor(ts[1]/self.step).astype(int)
        self.phase[i0:i1] = self.phase[i0]+linspace(0,ph, i1-i0)
        self.phase[i1:] = self.phase[i1:] + ph

    def os_ramp_by(self, ts, chg):
        i0 = floor(ts[0]/self.step).astype(int)
        i1 = floor(ts[1]/self.step).astype(int)
        self.offset[i0:i1] = linspace(self.offset[i0], self.offset[i1]+chg, i1-i0)
        self.offset[i1:] += chg

    def os_ramp_to(self, ts, targ):
        i0 = floor(ts[0]/self.step).astype(int)
        i1 = floor(ts[1]/self.step).astype(int)
        self.offset[i0:i1] = linspace(self.offset[i0],targ, i1-i0)
        self.amp[i1:] = targ

    def f_shift_by(self, t0, dfreq):
        i0 = floor(t0/self.step).astype(int)
        self.phase[i0:] = self.phase[i0:]+2*pi*dfreq*(self.time[i0:] - self.time[i0])

    def f_shift_to(self, t0, nfreq):
        i0 = floor(t0/self.step).astype(int)
        self.phase[i0:] = self.phase[i0]+2*pi*(nfreq-self.freq)*(self.time[i0:] - self.time[i0])

    def amp_ramp_to(self, ts, targ):
        i0 = floor(ts[0]/self.step).astype(int)
        i1 = floor(ts[1]/self.step).astype(int)
        self.amp[i0:i1] = linspace(self.amp[i0],targ, i1-i0)
        self.amp[i1:] = targ

    def amp_ramp_by(self, ts, chg):
        i0 = floor(ts[0]/self.step).astype(int)
        i1 = floor(ts[1]/self.step).astype(int)
        self.amp[i0:i1] = linspace(self.amp[i0],self.amp[i1]+chg, i1-i0)
        self.amp[i1:] += chg

    def phase_mod(self, ts, freq, amp, mod_phase):
        i0 = floor(ts[0]/self.step).astype(int)
        i1 = floor(ts[1]/self.step).astype(int)
        self.phase[i0:i1] = self.phase[i0] + amp*sin(2*pi*freq*self.time[i0:i1]+mod_phase)

    def scale(self, factor):
        self.amp = self.amp*factor

    def plot(self, wfm=False):
        if wfm:
            fig, ax1 = plt.subplots(figsize=(9, 1))
            ax1.plot(self.time*1e9, self.wfm)
            ax1.set_xlabel("time (ns)")
        else:
            #camp, cphase = "#1f77b4", "#ff7f0e"
            camp, cphase = "red", "green"
            fig, ax1 = plt.subplots(figsize=(9, 1))
            pamp = ax1.plot(self.time*1e9, self.amp, color=camp)
            ax1.set_ylabel("Mag", color=camp)
            ax1.tick_params(axis="y", labelcolor=camp)

            ax2 = ax1.twinx()
            ax2.plot(self.time*1e9, 180*self.phase/pi, color=cphase)
            ax2.set_ylabel("Ph (deg)", color=cphase)
            ax2.tick_params(axis="y", labelcolor=cphase)

            ax2.set_xlabel("time (ns)")
            plt.title(f"{self.name}, f0 = {self.freq/1e9:.2f} GHz")

    def _string(self):
        return(f"pwl(tm {self.name})")

    def _wfm(self):
        return self.amp*sin(2*pi*self.freq*self.time+self.phase)+self.offset

    wfm = property(fget=_wfm)
    wr_string = property(fget=_string)

def waveforms_to_file(wfms, filename = "waveforms.raw"):
    # All wafevorms are assumed to use the same time vector (sim time and step)
    wr_vars = [Variable("time", "", values=wfms[0].time)]
    for idx, wfm in ennumerate(wfms):
        if wfm.name is None:
            name = f"wfm{idx:.0d}"
        else:
            name = wfm.name
        wr_vars.append(Variable(name, "", values=wfm.wfm))
    write_rawfile(filename, wr_vars)
