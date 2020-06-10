

import numpy as np
from cpymad.madx import Madx
import sixtracklib as pystlib
import pysixtrack
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.colors as mcolors
#get_ipython().run_line_magic('matplotlib', 'inline')
from PyHEADTAIL.trackers.rf_bucket import RFBucket



import seaborn as sns
sns.set_context('talk', font_scale=1.2, rc={'lines.linewidth': 3})
sns.set_style('whitegrid',
              {'grid.linestyle': ':', 'grid.color': 'red', 'axes.edgecolor': '0.5',
               'axes.linewidth': 1.2, 'legend.frameon': True})




from PyHEADTAIL.particles import generators

from PyHEADTAIL.particles.generators import (
    generate_Gaussian6DTwiss, gaussian2D, 
    cut_distribution, make_is_accepted_within_n_sigma)




from scipy.constants import e, m_p, c

from scipy.constants import physical_constants

nmass = physical_constants['atomic mass constant energy equivalent in MeV'][0] * 1e-3


class Runner(object):
    
    A = 238
    Q = 28

    Ekin_per_nucleon = 0.2e9 # in eV
    
    epsx_rms_fin = 35e-6 / 4 # geometrical emittances
    epsy_rms_fin = 15e-6 / 4

    limit_n_rms_x = 2
    limit_n_rms_y = 2
    limit_n_rms_z = 3.4

    sig_z = 58 / 4. # in m
    sig_dp = 0.5e-3
    
    def __init__(self, nturns=20000, npart=1000):
        self.nturns = nturns
        self.npart = npart
        
        mass = self.A * nmass * 1e9 * e / c**2 # in kg
        charge = self.Q * e # in Coul
        Ekin = self.Ekin_per_nucleon *self.A
        p0c = np.sqrt(Ekin**2 + 2*Ekin*mass/e * c**2) # in eV
        Etot = np.sqrt(p0c**2 + (mass/e)**2 * c**4) * 1e-9 # in GeV
        p0 = p0c / c * e # in SI units
        gamma = np.sqrt(1 + (p0 / (mass * c))**2)
        beta = np.sqrt(1 - gamma**-2)
        
        self.beta = beta
        self.gamma = gamma
        self.p0 = p0
        self.Etot = Etot
        self.p0c = p0c
        self.charge = charge
        self.mass = mass
        
        epsx_gauss = self.epsx_rms_fin * 1.43
        epsy_gauss = self.epsy_rms_fin * 1.41

        self.epsn_x = epsx_gauss * beta * gamma
        self.epsn_y = epsy_gauss * beta * gamma

        self.sig_z = self.sig_z * 1.22
        self.sig_dp = self.sig_dp * 1.22

        self.beta_z = self.sig_z / self.sig_dp

        self.madx = Madx()
        self.madx.options.echo = False
        self.madx.options.warn = False
        self.madx.options.info = False
        
    def prepareMaxSimple(self):
        # prepare madx
    
        self.madx.command.beam(particle='ion', mass=self.A*nmass, charge=self.Q, energy=self.Etot)
        self.madx.call(file="SIS100_RF_220618_9slices.thin.seq")
        self.madx.use(sequence='sis100ring')
        
        twiss = self.madx.twiss()
        
        return twiss
        
        
        
    def prepareMax(self, qx, qy):
        # prepare madx
    
        self.madx.command.beam(particle='ion', mass=self.A*nmass, charge=self.Q, energy=self.Etot)
        self.madx.call(file="SIS100_RF_220618_9slices.thin.seq")
        self.madx.use(sequence='sis100ring')

        self.madx.input('''
        match, sequence=SIS100RING;
        global, sequence=SIS100RING, q1={qx}, q2={qy};
        vary, name=kqf, step=0.00001;
        vary, name=kqd, step=0.00001;
        lmdif, calls=500, tolerance=1.0e-10;
        endmatch;
        '''.format(qx=qx, qy=qy))
    
        
        twiss = self.madx.twiss()
    
        self.madx.input('cavity_voltage = 58.2/1000/number_cavities;')
    
        return twiss
        
    def setup_pysixtrack(self):
    
            seqname = 'sis100ring'
            sis100 = getattr(self.madx.sequence, seqname)
        
            pysixtrack_elements = pysixtrack.Line.from_madx_sequence(
                self.madx.sequence.sis100ring, exact_drift=True, install_apertures=True)

            pysixtrack_elements.remove_zero_length_drifts(inplace=True);
            pysixtrack_elements.merge_consecutive_drifts(inplace=True);
        
            return pysixtrack_elements
            
    def prepareDis(self, twiss, closed_orbit):
            
        if closed_orbit is not None:
            x_co = twiss[0]['x']
            y_co = twiss[0]['y']
        else:
            x_co = 0
            y_co = 0
    
        np.random.seed(0)
        D_x_0 = twiss[0]['dx'] * self.beta
        D_y_0 = twiss[0]['dy'] * self.beta

        Dp_x_0 = twiss[0]['dpx'] * self.beta
        Dp_y_0 = twiss[0]['dpy'] * self.beta
    
        bx_0 = twiss[0]['betx']
        by_0 = twiss[0]['bety']

        s0 = twiss[-1]['s']
        circumference = s0

        alfx_0 = twiss[0]['alfx']
        alfy_0 = twiss[0]['alfy']

        pyht_beam = generators.generate_Gaussian6DTwiss(
            self.npart, 1, self.charge, self.mass, s0, self.gamma,
            alfx_0, alfy_0, bx_0, by_0,
            1, self.epsn_x, self.epsn_y, 1,
            dispersion_x=None,
            dispersion_y=None,
            limit_n_rms_x=self.limit_n_rms_x**2, limit_n_rms_y=self.limit_n_rms_y**2, 
            limit_n_rms_z=self.limit_n_rms_z**2,
                )

        distribution_z_uncut = generators.gaussian2D(self.sig_z**2)
        is_accepted = generators.make_is_accepted_within_n_sigma(
            epsn_rms=self.sig_z,
            limit_n_rms=2.5,
        )
        distribution_z_cut = generators.cut_distribution(distribution_z_uncut, is_accepted)

        z, dp = distribution_z_cut(self.npart)
        pyht_beam.z, pyht_beam.dp = z, dp / self.beta_z

        # recentre on 0 to avoid dipolar motion:
        pyht_beam.x -= pyht_beam.mean_x()
        pyht_beam.xp -= pyht_beam.mean_xp()
        pyht_beam.y -= pyht_beam.mean_y()
        pyht_beam.yp -= pyht_beam.mean_yp()
        pyht_beam.z -= pyht_beam.mean_z()
        pyht_beam.dp -= pyht_beam.mean_dp()

        # PyHT generates around 0, need to offset with closed orbit:
        pyht_beam.x += x_co
        pyht_beam.y += y_co
        # add dispersive contribution to coordinates:
        pyht_beam.x += D_x_0 * pyht_beam.dp
        pyht_beam.y += D_y_0 * pyht_beam.dp
        # also need to add D'_{x,y} to momenta:
        pyht_beam.xp += Dp_x_0 * pyht_beam.dp
        pyht_beam.yp += Dp_y_0 * pyht_beam.dp
    
        return pyht_beam
        
    def setup_sixtracklib(self, pysixtrack_elements, pyht_beam):
    
        elements = pystlib.Elements.from_line(pysixtrack_elements)
        elements.BeamMonitor(num_stores=self.nturns);
        particles = pystlib.Particles.from_ref(self.npart, p0c=self.p0c, mass0 = self.A*nmass*1e9, q0 = self.Q)
    
        particles.x[:] = pyht_beam.x
        particles.px[:] = pyht_beam.xp
        particles.y[:] = pyht_beam.y
        particles.py[:] = pyht_beam.yp
        particles.zeta[:] = pyht_beam.z
        particles.delta[:] = pyht_beam.dp

        particles.rpp[:] = 1. / (pyht_beam.dp + 1)

        restmass = self.mass * c**2
        restmass_sq = restmass**2
        E0 = np.sqrt((self.p0 * c)**2 + restmass_sq)
        p = self.p0 * (1 + pyht_beam.dp)
        E = np.sqrt((p * c)**2 + restmass_sq)
        particles.psigma[:] = (E - E0) / (self.beta * self.p0 * c)

        gammai = E / restmass
        betai = np.sqrt(1 - 1. / (gammai * gammai))
        particles.rvv[:] = betai / self.beta
    
        ### prepare trackjob in SixTrackLib
        job = pystlib.TrackJob(elements, particles)
    
        return job



    def plotDis(self, pyht_beam):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        plt.sca(ax[0])
        plt.title('horizontal')
        plt.scatter(pyht_beam.x[::100] * 1e3, pyht_beam.xp[::100] * 1e3, s=10, marker='.')
        plt.xlim(1.1*pyht_beam.x.min() * 1e3, 1.1*pyht_beam.x.max() * 1e3)
        plt.ylim(1.1*pyht_beam.xp.min() * 1e3, 1.1*pyht_beam.xp.max() * 1e3)
        plt.xlabel('$x$ [mm]')
        plt.ylabel("$x'$ [mrad]")

        plt.sca(ax[1])
        plt.title('vertical')
        plt.scatter(pyht_beam.y[::100] * 1e3, pyht_beam.yp[::100] * 1e3, s=10, marker='.')
        plt.xlim(1.1*pyht_beam.y.min() * 1e3, 1.1*pyht_beam.y.max() * 1e3)
        plt.ylim(1.1*pyht_beam.yp.min() * 1e3, 1.1*pyht_beam.yp.max() * 1e3)
        plt.xlabel('$y$ [mm]')
        plt.ylabel("$y'$ [mrad]")

        plt.sca(ax[2])
        plt.title('longitudinal')
        plt.scatter(pyht_beam.z[::100], pyht_beam.dp[::100] * 1e3, s=10, marker='.')
        plt.xlabel('$z$ [m]')
        plt.ylabel(r"$\Delta p/p_0'$ [$10^{-3}$]")
        plt.tight_layout()
        plt.savefig('phasespace.png')



    def tuneFFT(self, x, y, twiss):
    
        q1mad = twiss.summary['q1']
        q2mad = twiss.summary['q2']
    
        ff = np.linspace(0, 0.5, self.nturns // 2 + 1)
        xf = abs(np.fft.rfft(x))
        q1st = ff[xf.argmax()]
    
        yf = abs(np.fft.rfft(y))
        q2st = ff[yf.argmax()] 
        #
        q1madFrac = q1mad % 1
        q1madFrac = q1madFrac if q1madFrac < 0.5 else 1 - q1madFrac
    
        q2madFrac = q2mad % 1
        q2madFrac = q2madFrac if q2madFrac < 0.5 else 1 - q2madFrac
        #
    
        #
        print('horizontal:', q1mad, round(1 - q1st,2), q1madFrac - q1st)
        print('vertical:', q2mad, round(1 - q2st,2), q2madFrac - q2st)
    
        return 1 - q1st, 1- q2st
    

    def setup_sixtracklib_fft(self, pysixtrack_elements):
        elements = pystlib.Elements.from_line(pysixtrack_elements)

        elements.BeamMonitor(num_stores=self.nturns);

        particles = pystlib.Particles.from_ref(self.npart, p0c=self.p0c)

        particles.x += np.linspace(0, 1e-6, self.npart)
        particles.y += np.linspace(0, 1e-6, self.npart)

        job = pystlib.TrackJob(elements, particles)
    
        return job


    def getData(self, job):
        x = job.output.particles[0].x[1::self.npart]
        y = job.output.particles[0].y[1::self.npart]
        #self.madx.exit()

    

if __name__ == '__main__':
    
    nturns0 = 200
    npart0 = int(1e3)
    closed_orbit = None
    
    simulation = Runner(nturns=nturns0, npart=npart0)
    
    twiss = simulation.prepareMaxSimple()
    
    pyht_beam = simulation.prepareDis(twiss, closed_orbit)
    simulation.plotDis(pyht_beam)
    
    qy_list = [18.73, 18.74]
    qx_list = [18.84, 18.85]

    tune_fft1 = []
    tune_fft2 = []

    for qx in qx_list:
        for qy in qy_list:    
        
            twiss = simulation.prepareMax(qx, qy)
        
            pysixtrack_elements = simulation.setup_pysixtrack()
        
            pyht_beam = simulation.prepareDis(twiss, closed_orbit)
            job =  simulation.setup_sixtracklib(pysixtrack_elements, pyht_beam)
        
            # random beam for tune check with fft
            #job = setup_sixtracklib_fft(pysixtrack_elements, nturns, npart )
        
            #job = pystlib.TrackJob(elements, particles, device='opencl:0.0')

            job.track_until(nturns0)


            job.collect()

            x = job.output.particles[0].x[1::npart0]
            y = job.output.particles[0].y[1::npart0]

            q1st, q2st = simulation.tuneFFT(x, y, twiss)

            tune_fft1.append(round(q1st,2))
            tune_fft2.append(round(q2st,2))
    


    print(tune_fft1, tune_fft2)
    