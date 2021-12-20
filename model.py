import os
import pickle

import numpy as np
from astropy import units as u
import astropy.constants as const
import scipy.stats as st
from scipy.interpolate import interp1d, RegularGridInterpolator, LinearNDInterpolator
from scipy.integrate import trapz, cumtrapz

import matplotlib.pyplot as plt
from labellines import labelLine, labelLines
from tqdm.notebook import tqdm

from astropy.cosmology import FlatLambdaCDM, z_at_value
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import xspec
from pyphot import astropy as pyphot

from numba import njit

lib = pyphot.get_library()

# https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node205.html#optxagnf
xspec.Xset.allowPrompting = False

def f_host_model(z, seed=None):
    np.random.seed(seed)
    log_f_host = -370.54432837*z**2 + 36.62199154*z -1.64364977 + np.random.normal(0, 0.3, size=len(z))
    return np.clip(10**log_f_host, 0, 1)

def g_minus_r_model(M_stellar, seed=None):
    np.random.seed(seed)
    x = np.log10(M_stellar) - 10
    return 0.03177466*x**2 + 0.25283705*x + 0.72109552 + np.random.normal(0, 0.3, size=len(M_stellar))

def calc_sigma_var(mag, magerr):
    
    N = np.shape(mag)[0] # Number of light curves
    nu = np.shape(mag)[1] - 1
    
    wt = 1/magerr**2
    m0 = np.sum(mag*wt, axis=1)/np.sum(wt, axis=1)
    m0 = np.array([m0]*(nu+1)).T # Reshape
        
    chisq = 1/nu*np.sum((mag - m0)**2*wt, axis=1)
    
    p = st.chi2.sf(chisq, nu) #1 - cdf
    
    log_p = np.log(p)
    
    sigma_var = np.zeros_like(p)
    
    mask_small = (log_p > -36)
    
    sigma_var[mask_small] = st.norm.ppf(1 - p[mask_small]/2)
    sigma_var[~mask_small] = np.sqrt(np.log(2/np.pi) - 2*np.log(8.2) - 2*log_p[~mask_small])
    
    return sigma_var


def f_occ_Bellovary19(M_star):
    """
    Heavy seed scenario
    """
    f = np.array([0.03, 0.06, 0.16, 0.2, 0.78, 1.0, 1.0, 1.0, 1.0])
    x = np.array([4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
    f_interp = np.interp(np.log10(M_star), x, f)
    df_interp = np.interp(np.log10(M_star), x, 0.3*f)
    return np.clip(f_interp, 0, 1)
    #return np.clip(np.random.normal(f_interp, df_interp), 0, 1)

def ERDF(lambda_Edd, mass_bin='med', z=0.0):
    xi = 10**(0.03*(1 + z) - 1.57)
    delta2 = -0.03*(1 + z) + 2.27
    
    lambda_norm = 0 #-3.0
    
    if mass_bin == 'high':
        lambda_br = 10**(0.70*(1 + z) - 2.60 + lambda_norm)
        delta1 = -0.04*(1 + z) + 0.45
    elif mass_bin == 'med':
        lambda_br = 10**(0.68*(1 + z) - 2.05 + lambda_norm)
        delta1 = -0.12*(1 + z) - 1.00
    elif mass_bin == 'low':
        lambda_br = 10**(1.27*(1 + z) - 2.53 + lambda_norm) # >-2.53
        delta1 = 0.00*(1 + z) + 0.5 # <0.5
        
    erdf = xi * ((lambda_Edd/lambda_br)**delta1 + (lambda_Edd/lambda_br)**delta2)**-1
    return erdf

def _ERDF(lambda_Edd):
    xi = 1
    # Lbr = 10**38.1 lambda_br M_BH_br
    # 10^41.67 = 10^38.1 * 10^x * 10^10.66
    lambda_br = 10**(-1.84) #10**(-1.84) # + np.random.normal(0, np.mean([0.30, 0.37])))
    delta1 = 0.47 #+ np.random.normal(0, np.mean([0.20, 0.42]))
    delta2 = 2.53 #+ np.random.normal(0, np.mean([0.68, 0.38]))
    # https://ui.adsabs.harvard.edu/abs/2019ApJ...883..139S/abstract
    # What sets the break? Transfer from radiatively efficient to inefficient accretion?
    return xi * ((lambda_Edd/lambda_br)**delta1 + (lambda_Edd/lambda_br)**delta2)**-1 # dN / dlog lambda


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

def lambda_obs(L_bol):
    
    L_bol = (L_bol*u.erg/u.s).to(u.Lsun)
    
    a = np.random.normal(10.96, 0.06)
    b = np.random.normal(11.93, 0.01)
    c = np.random.normal(17.79, 0.10)
    
    # https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..73D/abstract
    sig = np.random.normal(0, 0.27)
    sgn = np.sign(sig)
    k_X = a*(1 + (np.log10(L_bol/(1*u.Lsun))/b)**c) + sgn*np.log10(np.abs(sig))
    L_X = L_bol/k_X
    L_X = L_X.to(u.erg/u.s)
    
    A = 0.5 # Must be 0.5 so the range is betwen 0 and 1
    l_0 = 43.89
    sigma_x = 0.46
    
    # https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.3550M/abstract
    # Add 0.2 dex scatter 
    l_x = np.log10(L_X.value)
    lambda_obs = A + 1/np.pi*np.arctan((l_0 - l_x)/sigma_x)
    
    return np.clip(np.random.normal(lambda_obs, 0.1), 0, 1)

@njit
def drw(t_obs, x, xmean, SFinf, E, N):
    for i in range(1, N):
        dt = t_obs[i,:] - t_obs[i - 1,:]
        x[i,:] = (x[i - 1,:] - dt * (x[i - 1,:] - xmean) + np.sqrt(2) * SFinf * E[i,:] * np.sqrt(dt))
    return x


def simulate_drw(t_rest, tau=300., z=2.0, xmean=0, SFinf=0.3):

    N = np.shape(t_rest)[0]
    ndraw = len(tau)
    
    # t_rest [N, ndraw]

    t_obs = t_rest * (1. + z) / tau
    
    x = np.zeros((N, ndraw))
    x[0,:] = np.random.normal(xmean, SFinf)
    E = np.random.normal(0, 1, (N, ndraw))
    
    return drw(t_obs, x, xmean, SFinf, E, N)


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
    cum_values = np.zeros(x.shape)
    cum_values[1:] = np.cumsum(y*dx)/np.sum(y*dx)
    inv_cdf = interp1d(cum_values, x, fill_value='extrapolate')
    if survival:
        n_samples = int(survival)
        print(n_samples)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def survival_sampling(y, survival, fill_value=np.nan):
    # Can't randomly sample, will muck up indicies
    n_samples = len(y)
    randp = np.random.rand(n_samples)
    mask_rand = (randp < survival)
    y_survive = np.full(y.shape, fill_value)
    y_survive[mask_rand] = y[mask_rand]
    return y_survive


class DemographicModel:

    
    def __init__(self):
        # argument for input of multiple occupation fractions
        self.pars = {}
        self.samples = {}
        

    def sample(self, nbins=10, nbootstrap=50, eta=1e4, zmax=0.1, ndraw_dim=1e7, omega=4*np.pi,
               seed_dict={'dc':(lambda x: np.ones_like(x)), 'popIII':f_occ_Bellovary19}):

        """
        See https://iopscience.iop.org/article/10.3847/1538-4357/aa803b/pdf

        nbins: Number of stellar mass bins 
        nbootstrap: Number of bootstrap samples (for observational uncertainties)
        eta: Understamping factor (each MC data point corresponds to eta real galaxies)
        zmax:
        ndraw_dim: 
        occ_dict: Dictonary of occupation fractions to use
        """
        
        pars = {'nbins':nbins, 'nbootstrap':nbootstrap, 'eta':eta}
        samples = {}
        
        dtype = np.float64
        
        ndraw_dim = int(ndraw_dim)
        pars['ndraw_dim'] = ndraw_dim
        pars['seed_dict'] = seed_dict
        
        pars['log_lambda_min'] = -7.0
        pars['log_lambda_max'] = 1.0
        
        pars['log_M_star_min'] = 4.5
        pars['log_M_star_max'] = 12.5
        
        pars['log_M_BH_min'] = 1.5
        pars['log_M_BH_max'] = 9.5

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
        M_star_ = np.logspace(pars['log_M_star_min'], pars['log_M_star_max'], nbins+1, dtype=dtype)*u.Msun
        dM_star = np.diff(M_star_)
        dlogM_star = np.diff(np.log10(M_star_.value))
        pars['M_star'] = M_star_[1:] + dM_star/2 # bins
        pars['M_star_'] = M_star_

        # 1. Assign number of draws
        d_c_min = 0.5*u.Mpc
        samples['zmax'] = zmax
        samples['zmin'] = z_at_value(cosmo.comoving_distance, d_c_min, zmin=-1e-4, zmax=zmax+1e-4) # 1e-8
        V = cosmo.comoving_volume(zmax)*omega/(4*np.pi)
        pars['V'] = V
        #d_c_min = cosmo.comoving_distance(samples['zmin'])
        d_c_samples = np.linspace(d_c_min.to(u.Mpc), cosmo.comoving_distance(zmax).to(u.Mpc), 100, dtype=dtype)
        z_samples = np.array([z_at_value(cosmo.comoving_distance, d_c, zmin=-1e-4, zmax=zmax+1e-4)
                              for d_c in d_c_samples])

        # 2. Draw from the stellar mass function
        samples['M_star_draw'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)*u.Msun
        samples['n_i_M'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)

        # 4. BH Mass Function
        M_BH_ = np.logspace(pars['log_M_BH_min'], pars['log_M_BH_max'], nbins+1, dtype=dtype)*u.Msun
        dM_BH = np.diff(M_BH_)
        dlogM_BH = np.diff(np.log10(M_BH_.value))
        pars['M_BH'] = M_BH_[1:] + dM_BH/2 # bins

        samples['z_draw'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)

        samples['M_BH_draw'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)*u.Msun
        # Occupation probabililty survival sampling
        for k, seed in enumerate(seed_dict.keys()):
            samples[f'M_BH_draw_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)*u.Msun
            samples[f'n_i_M_{seed}'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)

        # 5. Eddington ratio Function
        lambda_ = np.logspace(pars['log_lambda_min'], pars['log_lambda_max'], nbins+1, dtype=dtype)
        dlambda = np.diff(lambda_)
        dloglambda = np.diff(np.log10(lambda_))
        pars['lambda_Edd'] = lambda_[1:] + dlambda/2 # bins

        samples['lambda_draw'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)
        samples['n_i_Edd'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)

        # 6. AGN Luminosity Function
        L_ = np.logspace(34, 47, nbins+1, dtype=dtype)*u.erg/u.s
        dL = np.diff(L_)
        dlogL = np.diff(np.log10(L_.value))
        pars['L'] = L_[1:] + dL/2 # bins

        # Occupation probabililty survival sampling
        for k, seed in enumerate(seed_dict.keys()):
            samples[f'L_draw_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)*u.erg/u.s
            samples[f'n_i_L_{seed}'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)

        samples['ndraws'] = np.empty(nbootstrap, dtype=np.int)
        
        for j in tqdm(range(nbootstrap)):
            
            np.random.seed(j)

            # 2. Draw from stellar mass function
            phidM = np.exp(-pars['M_star']/M_br[j]) * (phi1[j]*(pars['M_star']/M_br[j])**alpha1[j] +
                                                        phi2[j]*(pars['M_star']/M_br[j])**alpha2[j]) * dM_star/M_br[j]
            # phi = dN / dlog M
            sf = V.to(u.Mpc**3).value / eta * trapz((phidM/dM_star).value, pars['M_star'].value) # Normalization
            # Try taking log of (phidM/dM_star).value and sampling from that?
            M_star_draw = inv_transform_sampling((phidM/dM_star).value, M_star_.value, survival=sf)*u.Msun
            ndraw = len(M_star_draw)
            samples['ndraws'][j] = ndraw
            samples['M_star_draw'][j,:ndraw] = M_star_draw
            samples['n_i_M'][j,:], _ = np.histogram(M_star_draw, bins=M_star_)

            # Assign redshifts
            dz = np.mean(np.diff(z_samples))
            z_draw = np.random.choice(z_samples, size=ndraw) + np.random.normal(0, dz, size=ndraw)
            z_draw = np.clip(z_draw, samples['zmin'], samples['zmax'])
            samples['z_draw'][j,:ndraw] = z_draw
            
            # 4. BH Mass Function
            M_BH_draw = 10**(alpha[j] + beta[j]*np.log10(M_star_draw/(1e11*u.Msun)) +
                             np.random.normal(0.0, 0.55, size=ndraw))*u.Msun
            samples['M_BH_draw'][j,:ndraw] = M_BH_draw
            
            # 5. Eddington ratio Function
            xi_hi = ERDF(pars['lambda_Edd'], mass_bin='high') # dN / dlog lambda
            xi_med = ERDF(pars['lambda_Edd'], mass_bin='med') # dN / dlog lambda
            xi_lo = ERDF(pars['lambda_Edd'], mass_bin='low') # dN / dlog lambda
             # Normalize total ERDF
            norm = trapz(xi_lo + xi_hi + xi_med, pars['lambda_Edd'])
            xi_hi = xi_hi/norm
            xi_med = xi_med/norm
            xi_lo = xi_lo/norm
            # Masks
            mask_lo = (M_star_draw < 1e10*u.Msun)
            mask_med = (M_star_draw > 1e10*u.Msun) & (M_star_draw < 1e11*u.Msun)
            mask_hi = (M_star_draw > 1e11*u.Msun)
            # Draw
            lambda_draw_hi = inv_transform_sampling(xi_lo/dlambda, lambda_, np.count_nonzero(mask_hi))
            lambda_draw_med = inv_transform_sampling(xi_lo/dlambda, lambda_, np.count_nonzero(mask_med)) ###
            lambda_draw_lo = inv_transform_sampling(xi_lo/dlambda, lambda_, np.count_nonzero(mask_lo))
            lambda_draw = np.full(ndraw, np.nan)
            lambda_draw[mask_lo] = lambda_draw_lo
            lambda_draw[mask_med] = lambda_draw_med
            lambda_draw[mask_hi] = lambda_draw_hi
            samples['lambda_draw'][j,:ndraw] = lambda_draw
            samples['n_i_Edd'][j,:], _ = np.histogram(lambda_draw, bins=lambda_)
            
            # Occupation probabililty survival sampling
            for k, seed in enumerate(seed_dict.keys()):
                p = seed_dict[f'{seed}'](M_star_draw.value)
                M_BH_draw_seed = survival_sampling(M_BH_draw.value, survival=p, fill_value=0.0)*u.Msun
                samples[f'M_BH_draw_{seed}'][j,:ndraw] = M_BH_draw_seed
                samples[f'n_i_M_{seed}'][j,:], _ = np.histogram(M_BH_draw_seed, bins=M_BH_)

                # 6. AGN Luminosity Function
                L_draw_seed = lambda_draw * 1.26e38 * M_BH_draw_seed.to(u.Msun).value * u.erg/u.s
                samples[f'L_draw_{seed}'][j,:ndraw] = L_draw_seed
                samples[f'n_i_L_{seed}'][j,:], _ = np.histogram(L_draw_seed, bins=L_)

        # Correct for numerical factor and save the results
        for key in samples.keys():
            if key.startswith('n_i'):
                samples[key] = samples[key].astype(np.float)*eta
        
        self.pars = pars
        self.samples = samples
        
        
    def sample_sed_grid(self, w0=1e-5, w1=1e8, band='SDSS_g', model_sed_name='optxagnf', nbins=8,
                       sed_pars={'bh_mass':1e8,'dist_c':30.0,'lambda_edd':np.log10(0.1),'spin':0,'r_cor':100,
                                 'log_r_out':-1,'kT_e':0.2,'tau':10,'gamma':1.8,'f_pl':0.25,'z':0.007,'norm':1}):
        
        pars = self.pars
        s = self.samples
        
        dtype = np.float64
        
        ndraw_dim = int(np.max(s['ndraws']))
        
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
        x = np.logspace(pars['log_M_BH_min'], pars['log_M_BH_max'], nbins)
        y = np.logspace(pars['log_lambda_min'], pars['log_lambda_max'], nbins)
        z = np.linspace(s['zmin'], s['zmax'], nbins)
        
        print(f'Creating SED grid in band {band}')
        
        # Get AGN luminosity in band and M_i(z=2)
        vget_AGN_flux = np.vectorize(get_AGN_flux)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij', sparse=True)
        _, _, L_band_model, _ = vget_AGN_flux(model_sed, M_BH=X, lambda_Edd=Y, z=Z, band=band)
        
        X, Y = np.meshgrid(x, y, indexing='ij', sparse=True)
        M_i_model, _, _, _ = vget_AGN_flux(model_sed, M_BH=X, lambda_Edd=Y, z=2.0, band='SDSS_i')
        
        self.samples[f'L_{band}_model'] = L_band_model*u.erg/u.s # Shape NxNxN
        self.samples['M_i_model'] = M_i_model # Shape NxNxN
        
        # Create interpolator objects
        fn_L_band_model = RegularGridInterpolator((np.log10(x), np.log10(y), z), np.log10(L_band_model),
                                                  bounds_error=False, fill_value=None)
        fn_M_i_model = RegularGridInterpolator((np.log10(x), np.log10(y)), M_i_model,
                                               bounds_error=False, fill_value=None)
                
        for k, seed in enumerate(pars['seed_dict']):
            
            # MUST initialize L_AGN to 0, so host-galaxy mag is finite even when unoccupied!
            self.samples[f'L_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], 0.0, dtype=dtype)*u.erg/u.s
            self.samples[f'M_i_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)
            
            print(f'Sampling SEDs with seeding mechanism {seed}')
        
            # Sample the grid at each source
            for j in tqdm(range(nbootstrap)):

                ndraw = int(s['ndraws'][j])
                
                mask_occ = s[f'M_BH_draw_{seed}'][j,:ndraw].value != 0.0
                
                xj = np.log10(s[f'M_BH_draw_{seed}'].value[j,:ndraw][mask_occ])
                yj = np.log10(s['lambda_draw'][j,:ndraw][mask_occ])
                zj = s['z_draw'][j,:ndraw][mask_occ]
                points_3 = np.array([xj, yj, zj]).T
                points_2 = np.array([xj, yj]).T
                # Need to set the luminosity to 0 if M_BH=0
                self.samples[f'L_{band}_{seed}'][j,:ndraw][mask_occ] = (10**fn_L_band_model(points_3))*u.erg/u.s
                self.samples[f'M_i_{seed}'][j,:ndraw][mask_occ] = fn_M_i_model(points_2)
    
    
    def sample_light_curves(self, t_obs, dt_min=10, band='SDSS_g', SFinf_small=1e-2):
        
        pars = self.pars
        s = self.samples
        dtype = np.float64
        
        ndraw_dim = int(np.max(s['ndraws']))
        nbootstrap = pars['nbootstrap']
        
        t_obs_dense = np.arange(np.min(t_obs), np.max(t_obs), dt_min)
        t_rest_dense = np.arange(np.min(t_obs), np.max(t_obs), dt_min)
        
        s['lc_t_obs'] = t_obs
                        
        for k, seed in enumerate(pars['seed_dict']):
            
            print(f'Sampling light curves with seeding mechanism {seed}')
            
            s[f'lc_{band}_{seed}_idx'] = np.full([nbootstrap, ndraw_dim], np.arange(ndraw_dim), dtype=dtype)
            s[f'm_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)
            s[f'm_host_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)
            s[f'SFinf_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)
            s[f'tau_RF_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)
            s[f'L_host_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)*u.erg/u.s

            for j in tqdm(range(nbootstrap)):

                ndraw = int(s['ndraws'][j])

                z = s['z_draw'][j,:ndraw]
                M_BH = s[f'M_BH_draw_{seed}'][j,:ndraw].value
                M_star = s['M_star_draw'][j,:ndraw]
                # Model data
                L_band_AGN = s[f'L_{band}_{seed}'][j,:ndraw]
                L_bol_AGN = s[f'L_draw_{seed}'][j,:ndraw]
                M_i_AGN = s[f'M_i_{seed}'][j,:ndraw]

                lambda_RF = lib[band].lpivot/(1 + z)

                # Use the host M/L ratio to get host galaxy luminosity
                if band == 'SDSS_g':
                    a = -1.030 # g
                    b = 2.053
                elif band == 'GROUND_COUSINS_R':
                    a = -0.840 # r
                    b = 1.654
                else:
                    print("Not supported")
                    
                # Distributions of colors, and aperture flux ratios
                np.random.seed(j)
                color_var = np.random.normal(0.0, 0.3, size=ndraw)
                g_minus_r = g_minus_r_model(M_star.value, seed=j)
                f_host = f_host_model(z, seed=j)
                
                L_band_host = f_host * (M_star/(1*u.Msun) / 10**(b*g_minus_r + a + color_var))*u.Lsun
                L_band_host = L_band_host.to(u.erg/u.s)
                s[f'L_host_{band}_{seed}'][j,:ndraw] = L_band_host

                d_L = cosmo.comoving_distance(z).to(u.cm)
                f_band = (L_band_host + L_band_AGN) / (4*np.pi*d_L**2)
                f_lambda_band = (f_band / lib[band].lpivot).to(u.erg/u.s/u.cm**2/u.AA)

                # Get apparent magnitude of AGN + host galaxy
                m_band = -2.5*np.log10(f_lambda_band.value) - lib[band].AB_zero_mag
                
                # Color correction
                if band == 'GROUND_COUSINS_R':
                    # http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php#Lupton2005
                    m_band = m_band - 0.1837*g_minus_r - 0.0971 # R = ...
                s[f'm_{band}_{seed}'][j,:ndraw] = m_band
                                
                ##### Host
                f_band = L_band_host / (4*np.pi*d_L**2)
                f_lambda_band = (f_band / lib[band].lpivot).to(u.erg/u.s/u.cm**2/u.AA)
                # Get apparent magnitude of just host galaxy
                m_band = -2.5*np.log10(f_lambda_band.value) - lib[band].AB_zero_mag
                s[f'm_host_{band}_{seed}'][j,:ndraw] = m_band
                
                # Draw SF_\infty and tau (rest-frame)
                SFinf = draw_SFinf(lambda_RF.to(u.AA).value, M_i_AGN, M_BH, size=ndraw)
                tau = draw_tau(lambda_RF.to(u.AA).value, M_i_AGN, M_BH, size=ndraw)

                # Host-galaxy dilution
                dL = L_band_AGN*np.log(10)/2.5*SFinf # The 1.3 is to normalize with the qsos
                SFinf = 2.5/np.log(10)*dL/(L_band_AGN + L_band_host)

                mask_small = (SFinf < SFinf_small) | (M_BH < 1e2)
                
                t_obs_dense_shaped = (np.array([t_obs_dense]*ndraw)[~mask_small]).T
                t_rest_dense_shaped = (np.array([t_rest_dense]*ndraw)[~mask_small]).T / (1 + z[~mask_small])

                s[f'SFinf_{band}_{seed}'][j,:ndraw] = SFinf
                s[f'tau_RF_{band}_{seed}'][j,:ndraw] = tau
                
                # Simulate light curves (clip tau to avoid numerical issues if tau << dt_min)
                lcs = simulate_drw(t_rest_dense_shaped, np.clip(tau, 2*dt_min, None)[~mask_small],
                                   z[~mask_small], m_band[~mask_small], SFinf[~mask_small]).T
                f = interp1d(t_obs_dense, lcs, fill_value='extrapolate')
                s[f'lc_{band}_{seed}_{j}'] = f(t_obs)
                s[f'lc_{band}_{seed}_idx'][j,ndraw:] = np.nan
                s[f'lc_{band}_{seed}_idx'][j,:ndraw][mask_small] = np.nan
            

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
            
        
    def plot(self, figsize=(3*6*0.9, 2*5*0.9), seed_colors=['b','m'], seed_markers=['s','o']):
    
        import matplotlib.ticker as ticker

        fig, axs = plt.subplots(2, 3, figsize=figsize)
        axs = axs.flatten()
        
        pars = self.pars
        samples = self.samples
        
        ndraw_dim = int(np.max(samples['ndraws']))
        
        V = pars['V']
        n_i_M = samples['n_i_M'][:,:ndraw_dim]
        n_i_Edd = samples['n_i_Edd'][:,:ndraw_dim]
        
        M_star_draw = samples['M_star_draw'][:,:ndraw_dim]
        M_BH_draw = samples['M_BH_draw'][:,:ndraw_dim]
        
        M_star_ = pars['M_star_']
        
        M_star = pars['M_star']
        M_BH = pars['M_BH']
        lambda_Edd = pars['lambda_Edd']
        L = pars['L']
        dlogM_star = np.diff(np.log10(M_star.value))[0]
        dlogM_BH = np.diff(np.log10(M_BH.value))[0]
        dloglambda = np.diff(np.log10(lambda_Edd))[0]
        dlambda = np.diff(lambda_Edd)
        dlogL = np.diff(np.log10(L.value))[0]
                
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

        # 3. BH occupation fraction
        nbootstrap = pars['nbootstrap']

        for k, seed in enumerate(pars['seed_dict']):
            
            f = np.zeros([nbootstrap, len(M_star)])
            for j in range(nbootstrap):
                f[j,:] = pars['seed_dict'][f'{seed}'](M_star.value)
                
            axs[1].fill_between(M_star, np.percentile(f, 16, axis=0), np.percentile(f, 84, axis=0),
                                color=seed_colors[k], alpha=0.5)
            axs[1].scatter(M_star, np.nanmean(f, axis=0), lw=3, color=seed_colors[k], marker=seed_markers[k])
        
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

        # BH mass function
        for k, seed in enumerate(pars['seed_dict']):
            axs[3].fill_between(M_BH, np.nanpercentile(samples[f'n_i_M_{seed}'][:,:ndraw_dim], 16, axis=0)/dlogM_BH/V,
                                np.nanpercentile(samples[f'n_i_M_{seed}'][:,:ndraw_dim], 84, axis=0)/dlogM_BH/V, color=seed_colors[k], alpha=0.5)
            axs[3].scatter(M_BH, np.nanmean(samples[f'n_i_M_{seed}'][:,:ndraw_dim], axis=0)/dlogM_BH/V,
                           lw=3, color=seed_colors[k], marker=seed_markers[k])

        axs[3].set_xlabel(r'$M_{\rm{BH}}\ (M_{\odot})$', fontsize=18)
        axs[3].set_ylabel(r'$\phi(M_{\rm{BH}})$ (dex$^{-1}$ Mpc$^{-3}$)', fontsize=18)
        axs[3].set_xscale('log')
        axs[3].set_yscale('log')
        axs[3].set_xlim([1e2, 1e8])
        axs[3].set_ylim([1e-4, 1e0])

        # 5. Eddington ratio distribution (ERDF) function
        norm = np.sum(np.nanmean(n_i_Edd, axis=0)*dloglambda)
        axs[4].scatter(lambda_Edd, np.nanmean(n_i_Edd, axis=0)/dloglambda/norm, lw=3, color='k')
        norm1 = np.sum(np.nanpercentile(n_i_Edd, 16, axis=0)*dloglambda)
        norm2 = np.sum(np.nanpercentile(n_i_Edd, 84, axis=0)*dloglambda)
        axs[4].fill_between(lambda_Edd, np.nanpercentile(n_i_Edd, 16, axis=0)/dloglambda/norm1,
                            np.nanpercentile(n_i_Edd, 84, axis=0)/dloglambda/norm2, color='k', alpha=0.5)

        axs[4].set_xlabel(r'$\lambda_{\rm{Edd}}$', fontsize=18)
        axs[4].set_ylabel(r'$\xi$ (dex$^{-1}$)', fontsize=18)
        axs[4].set_xscale('log')
        axs[4].set_yscale('log')
        #axs[4].set_xlim([10**pars['log_lambda_min'] + dlambda[0], 10**pars['log_lambda_max']])
        axs[4].set_xlim([1e-5, 10**pars['log_lambda_max']])
        #axs[4].set_ylim([1e-8, 1e1])

        # 6. AGN Luminosity Function
        for k, seed in enumerate(pars['seed_dict']):
            axs[5].scatter(L, np.nanmean(samples[f'n_i_L_{seed}'][:,:ndraw_dim], axis=0)/dlogL/V,
                           lw=3, color=seed_colors[k], marker=seed_markers[k])
            axs[5].fill_between(L, np.nanpercentile(samples[f'n_i_L_{seed}'][:,:ndraw_dim], 16, axis=0)/dlogL/V,
                            np.nanpercentile(samples[f'n_i_L_{seed}'][:,:ndraw_dim], 84, axis=0)/dlogL/V,
                                color=seed_colors[k], alpha=0.5)
        
        # Real data
        # https://ui.adsabs.harvard.edu/abs/2009A%26A...507..781S/exportcitation
        x = np.array([42.63,42.83,43.03,43.24,43.44,43.65,43.86,44.07,44.28,44.49,\
                      44.71,44.93,45.15,45.38,45.60,45.82,46.05,46.28,46.51])
        y = np.array([-2.12,-2.48,-2.52,-3.02,-3.16,-3.64,-3.80,-3.98,-4.24,-4.59,\
                      -5.02,-5.24,-5.93,-6.42,-6.77,-6.98,-7.45,-7.78,-8.34])
        
        def phi_shen(L):
            # z = 0.2
            L_star = ((10**11.275)*u.Lsun).to(u.erg/u.s).value
            phi_star = 10**-4.240
            y1 = 0.787
            y2 = 1.713
            return phi_star / ( (L/L_star)**y1 + (L/L_star)**y2 )
            
        axs[5].scatter(10**x, 10**y, c='r', marker='x')
        x = np.logspace(42.5, 47)
        #axs[5].scatter(x, phi_shen(x), c='r', marker='x')

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

            ax.tick_params('both', labelsize=18)
            ax.tick_params('both', labelsize=18)
            ax.tick_params(axis='both', which='both', direction='in')
            ax.tick_params(axis='both', which='major', length=6)
            ax.tick_params(axis='both', which='minor', length=3)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')

            ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
            #ax.yaxis.set_major_locator(ticker.LogLocator(base=10))


        fig.tight_layout()
        return fig