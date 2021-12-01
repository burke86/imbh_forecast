import os
import pickle

import numpy as np
from astropy import units as u
import astropy.constants as const
import scipy.stats as st
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import trapz, cumtrapz

import matplotlib.pyplot as plt
from labellines import labelLine, labelLines
from tqdm.notebook import tqdm

from astropy.cosmology import FlatLambdaCDM, z_at_value
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import xspec
from pyphot import astropy as pyphot

lib = pyphot.get_library()

# https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node205.html#optxagnf
xspec.Xset.allowPrompting = False


# Heavy seed scenario (TODO: What about wandering BHs?)
def f_occ_Bellovary19(M_star):
    f = [0.03, 0.06, 0.16, 0.2, 0.78, 1.0, 1.0, 1.0, 1.0]
    x = [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    f_interp = np.interp(np.log10(M_star), x, f)
    return np.clip(np.random.normal(f_interp, 0.3*f_interp), 0, 1)

def _ERDF(lambda_Edd):
    xi = 10**-1.65
    lambda_br = 10**(-1.84) #10**(-1.84) # + np.random.normal(0, np.mean([0.30, 0.37])))
    delta1 = 0.47 #0.47 #+ np.random.normal(0, np.mean([0.20, 0.42]))
    delta2 = 2.5 #2.53 #+ np.random.normal(0, np.mean([0.68, 0.38]))
    # https://ui.adsabs.harvard.edu/abs/2019ApJ...883..139S/abstract
    # What sets the break? Transfer from radiatively efficient to inefficient accretion?
    return xi * ((lambda_Edd/lambda_br)**delta1 + (lambda_Edd/lambda_br)**delta2)**-1 # dN / dlog lambda

ERDF = np.vectorize(_ERDF)


def get_AGN_flux(model_sed, M_BH=1e6, lambda_Edd=0.1, z=0.01, band='SDSS_g'):
    
    bandpass = lib[band]
    # Input redshift
    d_L = cosmo.luminosity_distance(z).to(u.cm)
    d_c = cosmo.comoving_distance(z)
    # Parameters
    pars = {'bh_mass':M_BH,'dist_c':d_c.to(u.Mpc).value,'lambda_edd':np.log10(lambda_Edd),
            'spin':0,'r_cor':100,'log_r_out':-1,'kT_e':0.2,'tau':10,'gamma':1.8,'f_pl':0.25,'z':z,'norm':1}
    
    # Get the SED
    model_sed.setPars(list(pars.values()))
    energies = model_sed.energies(0)[::-1]*u.keV # Why N-1?
    # RF -> Obs. frame
    wav = (energies[:-1]).to(u.nm, equivalencies=u.spectral())*(1 + z)
    dwav = np.diff((energies).to(u.AA, equivalencies=u.spectral())*(1 + z))
    
    # E N_E dlogE to nu f_nu
    dlogE = np.diff(np.log10(energies[::-1].value))
    sed = model_sed.values(0)[::-1]/dlogE * u.photon/u.cm**2/u.s
    nuf_nu = sed.to(u.erg/u.cm**2/u.s, equivalencies=u.spectral_density(wav))
    
    # Get L_band
    f_lambda = nuf_nu/wav
    f_lambda_band = bandpass.get_flux(wav, f_lambda)
    f_band = f_lambda_band * bandpass.lpivot
    L_AGN_band = f_band * 4*np.pi*d_L**2
    
    # Get absolute magnitude
    m_band = -2.5*np.log10(f_lambda_band.value) - bandpass.AB_zero_mag
    M_band = m_band - 5*np.log10(d_L.to(u.pc).value) + 5 # This is M_band(z)
    
    # Make sure the bolometric luminosities are roughly consistent
    #L_lambda = f_lambda*4*np.pi*d_L**2
    #L_bol_Shen = lambda_Edd*1.26e38*(M_BH.to(u.Msun).value)*u.erg/u.s
    #L_bol = np.trapz(L_lambda.to(u.erg/u.s/u.AA), x=wav.to(u.AA)).to(u.erg/u.s)
    
    return M_band, m_band, L_AGN_band.to(u.erg/u.s).value, f_band.to(u.erg/u.s/u.cm**2).value


def simulate_drw(t_rest, tau=300., z=2.0, xmean=0, SFinf=0.3):

    N = np.shape(t_rest)[0]
    ndraw = len(tau)
    
    # t_rest [N, ndraw]

    t_obs = t_rest * (1. + z) / tau
    
    x = np.zeros([N, ndraw])
    x[0,:] = np.random.normal(xmean, SFinf)
    E = np.random.normal(0, 1, [N, ndraw])
    
    for i in range(1, N):
        dt = t_obs[i,:] - t_obs[i - 1,:]
        x[i,:] = (x[i - 1,:] - dt * (x[i - 1,:] - xmean) + np.sqrt(2) * SFinf * E[i,:] * np.sqrt(dt))
    return x


def draw_SFinf(lambda_RF, M_i, M_BH, size=1, randomize=True):
    if randomize:
        A = np.random.normal(-0.479, 0.008, size=size)
        B = np.random.normal(-0.479, 0.005, size=size)
        C = np.random.normal(0.118, 0.003, size=size)
        D = np.random.normal(0.118, 0.008, size=size)
        SFinf = 10**(A + B*np.log10(lambda_RF/4000) + C*(M_i + 23) + 
                 D*np.log10(M_BH/1e9) + np.random.normal(0, 0.09, size=len(M_BH))) # Delta mag
    else:
        A = -0.479
        B = -0.479
        C = 0.118
        D = 0.118
    
    SFinf = 10**(A + B*np.log10(lambda_RF/4000) + C*(M_i + 23) + 
                 D*np.log10(M_BH/1e9)) # Delta mag
    return SFinf


def draw_tau(lambda_RF, M_i, M_BH, size=1, randomize=True):
    # tau
    if randomize:
        A = np.random.normal(2.0, 0.01, size=size)
        B = np.random.normal(0.17, 0.02, size=size)
        C = 0 #np.random.normal(0.03, 0.04, size=size)
        D = np.random.normal(0.38, 0.05, size=size)
        tau = 10**(A + B*np.log10(lambda_RF/4000) + C*(M_i + 23) + 
                 D*np.log10(M_BH/1e8) + np.random.normal(0, 0.09, size=size)) # days
    else:
        A = 2.0
        B = 0.17
        C = 0.03
        D = 0.38
    
        tau = 10**(A + B*np.log10(lambda_RF/4000) + C*(M_i + 23) + 
                   D*np.log10(M_BH/1e8)) # days
    return tau


def inv_transform_sampling(y, x, n_samples=1000, survival=False):
    # https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
    # https://en.wikipedia.org/wiki/Inverse_transform_sampling
    dx = np.diff(x)
    dlogx = np.diff(np.log10(x))
    cum_values = np.zeros(x.shape)
    cum_values[1:] = np.cumsum(y*dx)/np.sum(y*dx)
    inv_cdf = interp1d(cum_values, x, fill_value='extrapolate')
    if survival:
        n_samples = int(survival)
        print(n_samples)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def survival_sampling(y, survival):
    # Can't randomly sample, will muck up indicies
    n_samples = len(y)
    randp = np.random.rand(n_samples)
    mask_rand = (randp < survival)
    y_survive = np.full(y.shape, np.nan)
    y_survive[mask_rand] = y[mask_rand]
    return y_survive


class DemographicModel:

    
    def __init__(self):
        # argument for input of multiple occupation fractions
        self.pars = {}
        self.samples = {}
        

    def sample(self, nbins=10, nbootstrap=50, eta=1e4, zmax=0.1, ndraw_dim=1e7,
               occ_dict={'dc':(lambda x: np.ones_like(x))}):

        """
        See https://iopscience.iop.org/article/10.3847/1538-4357/aa803b/pdf

        nbins: Number of stellar mass bins 
        nbootstrap: Number of bootstrap samples (for observational uncertainties)
        eta: Understamping factor (each MC data point corresponds to eta real galaxies)
        occ_dict: Dictonary of occupation fractions to use
        """
        
        pars = {'nbins':nbins, 'nbootstrap':nbootstrap, 'eta':eta}
        samples = {}
        
        ndraw_dim = int(ndraw_dim)
        pars['ndraw_dim'] = ndraw_dim

        # Use galaxy mass function from https://ui.adsabs.harvard.edu/abs/2012MNRAS.421..621B/abstract
        M_br = np.random.normal(loc=10**10.66, scale=10**0.05, size=nbootstrap)*u.Msun
        phi1 = np.random.normal(loc=3.96*1e-3, scale=0.34*1e-3, size=nbootstrap)*u.Mpc**-3
        phi2 = np.random.normal(loc=0.79*1e-3, scale=0.23*1e-3, size=nbootstrap)*u.Mpc**-3
        alpha1 = np.random.normal(loc=-0.35, scale=0.18, size=nbootstrap)
        alpha2 = np.random.normal(loc=-1.47, scale=0.05, size=nbootstrap)

        # M_BH - M_star relation from Reines 2015
        alpha = np.random.normal(loc=7.45, scale=0.08, size=nbootstrap)
        beta = np.random.normal(loc=1.05, scale=0.11, size=nbootstrap)

        # Stellar Mass Function
        M_star_ = np.logspace(4.5, 12.5, nbins+1)*u.Msun
        dM_star = np.diff(M_star_)
        dlogM_star = np.diff(np.log10(M_star_.value))
        pars['M_star'] = M_star_[1:] + dM_star/2 # bins
        pars['M_star_'] = M_star_ ## ??

        # 1. Assign number of draws
        d_c_min = 0.5*u.Mpc
        samples['zmax'] = zmax
        samples['zmin'] = z_at_value(cosmo.comoving_distance, d_c_min, zmin=-0.1, zmax=zmax+0.1)
        V = cosmo.comoving_volume(zmax)
        pars['V'] = V
        d_c_samples = np.linspace(d_c_min, cosmo.comoving_distance(zmax).to(u.Mpc), 100)
        z_samples = np.array([z_at_value(cosmo.comoving_distance, d_c, zmin=-0.1, zmax=zmax+0.1)
                              for d_c in d_c_samples])

        # 2. Draw from the stellar mass function
        samples['M_star_draw'] = np.full([nbootstrap, ndraw_dim], np.nan)*u.Msun
        samples['M_star_draw_test'] = np.full([nbootstrap, ndraw_dim], np.nan)*u.Msun
        samples['n_i_M'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)

        # 3. BH occupation fraction
        f_popIII = np.ones_like(pars['M_star'].value) # light
        f_dc = f_occ_Bellovary19(pars['M_star'].value) # dN / dlog lambda_EDD heavy

        # 4. BH Mass Function
        M_BH_ = np.logspace(1.0, 9.5, nbins+1)*u.Msun
        dM_BH = np.diff(M_BH_)
        dlogM_BH = np.diff(np.log10(M_BH_.value))
        pars['M_BH'] = M_BH_[1:] + dM_BH/2 # bins

        samples['z_draw'] = np.full([nbootstrap, ndraw_dim], np.nan)

        samples['M_BH_draw'] = np.full([nbootstrap, ndraw_dim], np.nan)*u.Msun
        samples['M_BH_draw_dc'] = np.full([nbootstrap, ndraw_dim], np.nan)*u.Msun
        samples['M_BH_draw_popIII'] = np.full([nbootstrap, ndraw_dim], np.nan)*u.Msun
        samples['n_i_M_dc'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)
        samples['n_i_M_popIII'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)

        # 5. Eddington ratio Function
        lambda_ = np.logspace(-3.5, 1, nbins+1) # Needs to match Weigel et al. or \xi will need to be re-normalized
        dlambda = np.diff(lambda_)
        dloglambda = np.diff(np.log10(lambda_))
        pars['lambda_Edd'] = lambda_[1:] + dlambda/2 # bins

        samples['lambda_draw'] = np.full([nbootstrap, ndraw_dim], np.nan)
        samples['n_i_Edd'] = np.full([nbootstrap, nbins], np.nan)

        # 6. AGN Luminosity Function
        L_ = np.logspace(34, 47, nbins+1)*u.erg/u.s
        dL = np.diff(L_)
        dlogL = np.diff(np.log10(L_.value))
        pars['L'] = L_[1:] + dL/2 # bins

        samples['L_draw_popIII'] = np.full([nbootstrap, ndraw_dim], np.nan)*u.erg/u.s
        samples['L_draw_dc'] = np.full([nbootstrap, ndraw_dim], np.nan)*u.erg/u.s

        samples['n_i_L_popIII'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)
        samples['n_i_L_dc'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)

        samples['ndraws'] = np.empty(nbootstrap) ## TODO: Can we just combine these?
        
        for j in tqdm(range(nbootstrap)):

            # 2. Draw from stellar mass function
            phidM = np.exp(-pars['M_star']/M_br[j]) * (phi1[j]*(pars['M_star']/M_br[j])**alpha1[j] +
                                                        phi2[j]*(pars['M_star']/M_br[j])**alpha2[j]) * dM_star/M_br[j]
            # phi = dN / dlog M
            sf = V.to(u.Mpc**3).value / eta * trapz((phidM/dM_star).value, pars['M_star'].value) # Normalization
            M_star_draw = inv_transform_sampling((phidM/dM_star).value, M_star_.value, survival=sf)*u.Msun
            ndraw = len(M_star_draw)
            samples['ndraws'][j] = ndraw
            samples['M_star_draw'][j,:ndraw] = M_star_draw
            samples['n_i_M'][j,:], _ = np.histogram(samples['M_star_draw'][j,:ndraw], bins=M_star_)

            # Assign redshifts
            samples['z_draw'][j,:ndraw] = np.random.choice(z_samples, size=ndraw)
            
            # 4. BH Mass Function
            M_BH_draw = 10**(alpha[j] + beta[j]*np.log10(samples['M_star_draw'][j,:ndraw]/(1e11*u.Msun)) +
                             np.random.normal(0.0, 0.55, size=ndraw))*u.Msun
            samples['M_BH_draw'][j,:ndraw] = M_BH_draw

            # Occupation probabililty survival sampling
            p = f_occ_Bellovary19(M_star_draw.value)
            samples['M_BH_draw_dc'][j,:ndraw] = survival_sampling(M_BH_draw.value, survival=p)*u.Msun
            samples['n_i_M_dc'][j,:], _ = np.histogram(samples['M_BH_draw_dc'][j,:ndraw], bins=M_BH_)

            # Occupation probabililty
            p = np.ones_like(M_star_draw.value)
            samples['M_BH_draw_popIII'][j,:ndraw] = survival_sampling(M_BH_draw.value, survival=p)*u.Msun
            samples['n_i_M_popIII'][j,:], _ = np.histogram(samples['M_BH_draw_popIII'][j,:ndraw], bins=M_BH_)

            # 5. Eddington ratio Function
            xi = ERDF(pars['lambda_Edd'])/dlambda # dN / dlog lambda
            samples['lambda_draw'][j,:ndraw] = inv_transform_sampling(xi/dlambda, lambda_, ndraw)
            samples['n_i_Edd'][j,:], _ = np.histogram(samples['lambda_draw'][j,:ndraw], bins=lambda_)

            # 6. AGN Luminosity Function
            samples['L_draw_popIII'][j,:ndraw] = samples['lambda_draw'][j,:ndraw] * 1.26e38 * (samples['M_BH_draw_popIII'][j,:ndraw]/(1*u.Msun)) *u.erg/u.s
            samples['L_draw_dc'][j,:ndraw] = samples['lambda_draw'][j,:ndraw] * 1.26e38 * (samples['M_BH_draw_dc'][j,:ndraw]/(1*u.Msun)) *u.erg/u.s
            samples['n_i_L_popIII'][j,:], _ = np.histogram(samples['L_draw_popIII'][j,:ndraw], bins=L_)
            samples['n_i_L_dc'][j,:], _ = np.histogram(samples['L_draw_dc'][j,:ndraw], bins=L_)            

        # Correct for numerical factor and save the results
        for key in samples.keys():
            if key.startswith('n_i'):
                samples[key] = samples[key].astype(np.float)*eta
        
        self.pars = pars
        self.samples = samples
        
        
    def sample_sed_grid(self, w0=1e-5, w1=1e8, band='SDSS_g', model_sed_name='optxagnf',
                       sed_pars={'bh_mass':1e8,'dist_c':30.0,'lambda_edd':np.log10(0.1),'spin':0,'r_cor':100,
                                 'log_r_out':-1,'kT_e':0.2,'tau':10,'gamma':1.8,'f_pl':0.25,'z':0.007,'norm':1}):
        
        pars = self.pars
        s = self.samples
        
        ndraw = pars['ndraw_dim']
        
        self.model_sed_name = model_sed_name
        self.w0 = w0
        self.w1 = w1
        self.sed_pars = sed_pars
        
        nbootstrap = pars['nbootstrap']
        
        print('Setting up model SED')
        
        # Initialize the model SED
        e0 = (w1*u.nm).to(u.keV, equivalencies=u.spectral())
        e1 = (w0*u.nm).to(u.keV, equivalencies=u.spectral())
        xspec.AllModels.setEnergies(f"{e0.value} {e1.value} 1000 log")
        model_sed = xspec.Model(model_sed_name)
        
        # Initalize the grid
        x = pars['M_BH'].value
        y = pars['lambda_Edd']
        z = np.linspace(s['zmin'], s['zmax'], pars['nbins'])
        
        print(f'Creating SED grid in band {band}')
        
        # Get AGN luminosity in band and M_i(z=2)
        vget_AGN_flux = np.vectorize(get_AGN_flux)
        
        X, Y, Z = np.meshgrid(x, y, z)
        _, _, L_band_model, _ = vget_AGN_flux(model_sed, M_BH=X, lambda_Edd=Y, z=Z, band=band)
        
        X, Y = np.meshgrid(x, y)
        M_i_model, _, _, _ = vget_AGN_flux(model_sed, M_BH=X, lambda_Edd=Y, z=2.0, band='SDSS_i')
        
        self.samples[f'L_{band}'] = np.full([nbootstrap, ndraw], np.nan)*u.erg/u.s
        self.samples['M_i'] = np.full([nbootstrap, ndraw], np.nan)
        self.samples[f'L_{band}_model'] = L_band_model*u.erg/u.s # Shape NxNxN
        self.samples['M_i_model'] = M_i_model # Shape NxNxN
        
        # Create interpolator objects
        fn_L_band_model = RegularGridInterpolator((x, y, z), L_band_model, bounds_error=False, fill_value=None)
        fn_M_i_model = RegularGridInterpolator((x, y), M_i_model, bounds_error=False, fill_value=None)
        
        print(f'Sampling SEDs')
        
        # Sample the grid at each source
        for j in tqdm(range(nbootstrap)):
            
            ndraw = int(s['ndraws'][j])
            
            xj = s['M_BH_draw'].value[j,:ndraw]
            yj = s['lambda_draw'][j,:ndraw]
            zj = s['z_draw'][j,:ndraw]
            points_3 = np.array([xj, yj, zj]).T
            points_2 = np.array([xj, yj]).T
            
            self.samples[f'L_{band}'][j,:ndraw] = fn_L_band_model(points_3)*u.erg/u.s
            self.samples['M_i'][j,:ndraw] = fn_M_i_model(points_2)
    
    
    def sample_light_curves(self, t_obs, dt_min=10, band='SDSS_g', f_host=0.5,  g_minus_i=0.8, mag_lim=np.inf):
        
        pars = self.pars
        s = self.samples
        
        ndraw = pars['ndraw_dim']
        nbootstrap = pars['nbootstrap']
            
        s[f'lc_{band}'] = np.full([nbootstrap, ndraw, len(t_obs)], np.nan)
        s[f'm_{band}'] = np.full([nbootstrap, ndraw], np.nan)
        s['lc_t_obs'] = t_obs
        s[f'SFinf_{band}'] = np.full([nbootstrap, ndraw], np.nan)
        s[f'tau_RF_{band}'] = np.full([nbootstrap, ndraw], np.nan)
        s[f'L_host_{band}'] = np.full([nbootstrap, ndraw], np.nan)*u.erg/u.s
                
        for j in tqdm(range(nbootstrap)):
            
            ndraw = int(s['ndraws'][j])
            
            z = s['z_draw'][j,:ndraw]
            M_BH = s['M_BH_draw'][j,:ndraw].value
            L_band_AGN = s[f'L_{band}'][j,:ndraw]
            M_i_AGN = s[f'M_i'][j,:ndraw]
            
            lambda_RF = lib[band].lpivot/(1 + z)
            
            # Use the host M/L ratio to get host galaxy luminosity
            # TODO: Assumes g-band (need dict of coefficients)
            M_stellar = 1e11*u.Msun * 10**((np.log10(M_BH + np.random.normal(0.0, 0.5, size=ndraw)) - 7.45)/1.05)
            L_band_host = f_host * M_stellar/(1*u.Msun) / 10**(2.053*g_minus_i - 1.030 + np.random.normal(0.0, 0.3, size=ndraw))*u.Lsun
            L_band_host = L_band_host.to(u.erg/u.s)
            s[f'L_host_{band}'][j,:ndraw] = L_band_host
            
            d_L = cosmo.comoving_distance(z).to(u.cm)
            f_band = (L_band_host + L_band_AGN) / (4*np.pi*d_L**2)
            f_lambda_band = (f_band / lib[band].lpivot).to(u.erg/u.s/u.cm**2/u.AA)

            # Get apparent magnitude of AGN + host galaxy
            m_band = -2.5*np.log10(f_lambda_band.value) - lib[band].AB_zero_mag
            s[f'm_{band}'][j,:ndraw] = m_band
                        
            ###
            # Draw SF_\infty and tau (rest-frame)
            SFinf = draw_SFinf(lambda_RF.to(u.AA).value, M_i_AGN, M_BH, size=ndraw)
            tau = draw_tau(lambda_RF.to(u.AA).value, M_i_AGN, M_BH, size=ndraw)

            # Host-galaxy dilution
            dL = L_band_AGN*np.log(10)/2.5*SFinf
            SFinf = 2.5/np.log(10)*dL/(L_band_AGN + L_band_host)
            
            t_obs_dense = np.arange(np.min(t_obs), np.max(t_obs), dt_min)
            t_obs_dense_shaped = np.array([t_obs_dense]*ndraw).T
            t_rest_dense_shaped = np.array([np.arange(np.min(t_obs), np.max(t_obs), dt_min)]*ndraw).T / (1 + z)
            
            s[f'SFinf_{band}'][j,:ndraw] = SFinf
            s[f'tau_RF_{band}'][j,:ndraw] = tau
            
            # Simulate light curves (clip tau to avoid numerical issues if tau << dt_min)
            lcs = simulate_drw(t_rest_dense_shaped, np.clip(tau, 2*dt_min, None), z, m_band, SFinf).T
            f = interp1d(t_obs_dense, lcs, fill_value='extrapolate')
            s[f'lc_{band}'][j,:ndraw,:] = f(t_obs)
            

    def save(self, filepath='model.pickle'):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file) 
            
        
    @staticmethod
    def load(filepath='model.pickle'):
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
            
            pars = model.pars
            s = model.samples
            
            return DemographicModel()
            
        
    def plot(self, figsize=(3*6*0.9, 2*5*0.9)):
    
        import matplotlib.ticker as ticker

        fig, axs = plt.subplots(2, 3, figsize=figsize)
        axs = axs.flatten()
        
        pars = self.pars
        samples = self.samples
        
        V = pars['V']
        n_i_M = samples['n_i_M']
        n_i_M_dc = samples['n_i_M_dc']
        n_i_M_popIII = samples['n_i_M_popIII']
        n_i_Edd = samples['n_i_Edd']
        n_i_L_popIII = samples['n_i_L_popIII']
        n_i_L_dc = samples['n_i_L_dc']
        
        M_star_draw = samples['M_star_draw']
        M_BH_draw = samples['M_BH_draw']
        
        M_star_ = pars['M_star_']
        
        M_star = pars['M_star']
        M_BH = pars['M_BH']
        lambda_Edd = pars['lambda_Edd']
        L = pars['L']
        dlogM_star = np.diff(np.log10(M_star.value))[0]
        dlogM_BH = np.diff(np.log10(M_BH.value))[0]
        dloglambda = np.diff(np.log10(lambda_Edd))[0]
        dlogL = np.diff(np.log10(L.value))[0]
        
        ## TODO: Generalize input
        f_popIII = np.ones_like(M_star) # light
        f_dc = f_occ_Bellovary19(M_star.value) # dN / dlog lambda_EDD heavy
        
        print(dlogM_star)
        
        # phi dM / dlogM
        axs[0].fill_between(M_star, np.nanpercentile(n_i_M, 16, axis=0)/dlogM_star/V,
                            np.nanpercentile(n_i_M, 84, axis=0)/dlogM_star/V, color='k', alpha=0.5)
        axs[0].scatter(M_star, np.nanmean(n_i_M, axis=0)/dlogM_star/V, lw=3, color='k')

        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel(r'$M_{\star}\ (M_{\odot})$', fontsize=18)
        axs[0].set_ylabel(r'$\phi(M_{\star})$ (dex$^{-1}$ Mpc$^{-3}$)', fontsize=18)
        axs[0].set_xlim([1e6, 1e12])
        axs[0].set_ylim([1e-4, 1e0])

        # Real data
        # https://ui.adsabs.harvard.edu/abs/2012MNRAS.421..621B
        x = np.array([6.25,6.75,7.10,7.30,7.50,7.70,7.90,8.10,8.30,8.50,8.70,8.90,9.10,9.30,9.50,9.70,9.90,\
                      10.1,10.3,10.5,10.7,10.9,11.1,11.3,11.5,11.7,11.9])
        y = np.array([31.1,18.1,17.9,43.1,31.6,34.8,27.3,28.3,23.5,19.2,18.0,14.3,10.2,9.59,7.42,6.21,5.71,5.51,5.48,5.12,\
                      3.55,2.41,1.27,0.33,0.042,0.021,0.042])
        axs[0].scatter(10**x, y*1e-3, c='r', marker='x')

        # 3. BH Occupation fraction
        axs[1].plot(M_star, f_popIII, lw=3, color='b', alpha=0.5)
        axs[1].scatter(M_star, f_popIII, lw=3, color='b')
        nbootstrap = pars['nbootstrap']
        f_dc = np.zeros([nbootstrap, len(M_star)])
        for j in range(nbootstrap):
            f_dc[j,:] = f_occ_Bellovary19(M_star.value)
        axs[1].fill_between(M_star, np.percentile(f_dc, 16, axis=0), np.percentile(f_dc, 84, axis=0), color='m', alpha=0.5)
        axs[1].scatter(M_star, np.nanmean(f_dc, axis=0), lw=3, color='m')
        
        axs[1].set_xlabel(r'$M_{\star}\ (M_{\odot})$', fontsize=18)
        axs[1].set_ylabel(r'$\lambda_{\rm{occ}}$', fontsize=18)
        axs[1].set_xscale('log')
        axs[1].set_xlim([1e6, 1e11])
        axs[1].set_ylim([0, 1.1])

        # M_BH - M_star
        bin_med, _, _ = st.binned_statistic(M_star_draw.flatten(), M_BH_draw.flatten(), np.nanmedian, bins=M_star_)
        bin_hi, _, _ = st.binned_statistic(M_star_draw.flatten(), M_BH_draw.flatten(), lambda x: np.nanpercentile(x, 84), bins=M_star_)
        bin_lo, _, _ = st.binned_statistic(M_star_draw.flatten(), M_BH_draw.flatten(), lambda x: np.nanpercentile(x, 16), bins=M_star_)

        axs[2].scatter(M_star, bin_med, lw=3, color='k')
        axs[2].fill_between(M_star, bin_hi, bin_lo, color='k', alpha=0.5)
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')
        axs[2].set_xlim([1e6, 1e11])
        axs[2].set_ylim([1e2, 1e8])
        axs[2].set_ylabel(r'$M_{\rm{BH}}\ (M_{\odot})$', fontsize=18)
        axs[2].set_xlabel(r'$M_{\rm{\star}}\ (M_{\odot})$', fontsize=18)

        # 
        axs[3].fill_between(M_BH, np.nanpercentile(n_i_M_popIII, 16, axis=0)/dlogM_BH/V,
                            np.nanpercentile(n_i_M_popIII, 84, axis=0)/dlogM_BH/V, color='b', alpha=0.5)
        axs[3].scatter(M_BH, np.nanmean(n_i_M_popIII, axis=0)/dlogM_BH/V, lw=3, color='b')

        axs[3].fill_between(M_BH, np.nanpercentile(n_i_M_dc, 16, axis=0)/dlogM_BH/V,
                            np.nanpercentile(n_i_M_dc, 84, axis=0)/dlogM_BH/V, color='m', alpha=0.5)
        axs[3].scatter(M_BH, np.nanmean(n_i_M_dc, axis=0)/dlogM_BH/V, lw=3, color='m')

        axs[3].set_xlabel(r'$M_{\rm{BH}}\ (M_{\odot})$', fontsize=18)
        axs[3].set_ylabel(r'$\phi(M_{\rm{BH}})$ (dex$^{-1}$ Mpc$^{-3}$)', fontsize=18)
        axs[3].set_xscale('log')
        axs[3].set_yscale('log')
        axs[3].set_xlim([1e2, 1e8])
        axs[3].set_ylim([1e-4, 1e0])

        # 5. Eddington ratio Function
        axs[4].scatter(lambda_Edd, np.nanmean(n_i_Edd, axis=0)/dloglambda, lw=3, color='k')
        axs[4].fill_between(lambda_Edd, np.nanpercentile(n_i_Edd, 16, axis=0)/dloglambda,
                            np.nanpercentile(n_i_Edd, 84, axis=0)/dloglambda, color='k', alpha=0.5)

        axs[4].set_xlabel(r'$\lambda_{\rm{Edd}}$', fontsize=18)
        axs[4].set_ylabel(r'$\xi$ (dex$^{-1}$)', fontsize=18)
        axs[4].set_xscale('log')
        axs[4].set_yscale('log')
        axs[4].set_xlim([1e-3, 1e0])
        #axs[4].set_ylim([1e-8, 1e1])

        # 6. AGN Luminosity Function
        axs[5].scatter(L, np.nanmean(n_i_L_popIII, axis=0)/dlogL/V, lw=3, color='b')
        axs[5].fill_between(L, np.nanpercentile(n_i_L_popIII, 16, axis=0)/dlogL/V,
                            np.nanpercentile(n_i_L_popIII, 84, axis=0)/dlogL/V, color='b', alpha=0.5)
        axs[5].scatter(L, np.nanmean(n_i_L_dc, axis=0)/dlogL/V, lw=3, color='m')
        axs[5].fill_between(L, np.nanpercentile(n_i_L_dc, 16, axis=0)/dlogL/V,
                            np.nanpercentile(n_i_L_dc, 84, axis=0)/dlogL/V, color='m', alpha=0.5)
        
        # Real data
        # https://ui.adsabs.harvard.edu/abs/2009A%26A...507..781S/exportcitation
        x = np.array([42.63,42.83,43.03,43.24,43.44,43.65,43.86,44.07,44.28,44.49,\
                      44.71,44.93,45.15,45.38,45.60,45.82,46.05,46.28,46.51])
        y = np.array([-2.12,-2.48,-2.52,-3.02,-3.16,-3.64,-3.80,-3.98,-4.24,-4.59,\
                      -5.02,-5.24,-5.93,-6.42,-6.77,-6.98,-7.45,-7.78,-8.34])
        axs[5].scatter(10**x, 10**y, c='r', marker='x')

        # Store variables for later
        axs[5].set_xlabel(r'$L_{\rm{bol}}$ (erg s$^{-1}$)', fontsize=18)
        axs[5].set_ylabel(r'$\phi(L_{\rm{bol}})$ (dex$^{-1}$ Mpc$^{-3}$)', fontsize=18)
        axs[5].set_xscale('log')
        axs[5].set_yscale('log')
        axs[5].set_xlim([1e36, 1e47])
        #axs[5].set_ylim([1e-4, 1e1])

        import string
        labels = list(string.ascii_lowercase)

        for i, ax in enumerate(axs):

            ax.text(0.02, 0.93, f'({labels[i]})', transform=ax.transAxes, fontsize=16, weight='bold', zorder=10)
            ax.tick_params(axis='x', which='major', pad=7)

            #ax.tick_params('both', labelsize=18)
            #ax.tick_params('both', labelsize=18)
            #ax.tick_params(axis='both', which='both', direction='in')
            #ax.tick_params(axis='both', which='major', length=6)
            #ax.tick_params(axis='both', which='minor', length=3)
            #ax.xaxis.set_ticks_position('both')
            #ax.yaxis.set_ticks_position('both')

            #ax.xaxis.set_major_locator(ticker.LogLocator(1))
            #ax.yaxis.set_major_locator(ticker.LogLocator(1))


        fig.tight_layout()
        return fig