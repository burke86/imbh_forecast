import sys, os
import pickle

from pathlib import Path
import re
import subprocess

import numpy as np
from astropy import units as u
import astropy.constants as const
import scipy.stats as st
from scipy.interpolate import interp1d, RegularGridInterpolator, LinearNDInterpolator
from numpy.polynomial.polynomial import polyval2d
from scipy.integrate import trapz, cumtrapz
from astropy.io import fits

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

def f_host_model(z, M_star, seed=None):
    np.random.seed(seed)
    
    log_M_star = np.log10(M_star)
    
    # Coefficients  a b c
    c5 = np.array([-0.2210982, 30.3864133, -5.6608222])
    c6 = np.array([-0.18302293, 49.3336501,  -8.01008124])
    c7 = np.array([-0.22146297, 32.87370419, -6.30751403])
    c8 = np.array([-0.30630442, 15.40596406, -3.78152073])
    c9 = np.array([-0.40381538, 13.43175193, -4.56761656])
    c10 = np.array([-0.36101119,  6.96839364, -1.58780549])
    
    # rms
    sig5 = 0.6536516
    sig6 = 0.49324566
    sig7 = 0.42940333
    sig8 = 0.33659458
    sig9 = 0.32550654
    sig10 = 0.28981143 # This is probably the most accurate
    
    cs = np.array([c5, c6, c7, c8, c9, c10])
    sigs = np.array([sig5, sig6, sig7, sig8, sig9, sig10])
    
    mask5 = log_M_star < 6
    mask6 = (log_M_star > 6) & (log_M_star < 7)
    mask7 = (log_M_star > 7) & (log_M_star < 8)
    mask8 = (log_M_star > 8) & (log_M_star < 9)
    mask9 = (log_M_star > 9) & (log_M_star < 10)
    mask10 = log_M_star > 10
    
    f_host = np.full_like(z, np.nan)
    for i, mask in enumerate([mask5, mask6, mask7, mask8, mask9, mask10]):
        x = z[mask] - cs[i][0]
        f_host[mask] = 1 - 1/(x**2 + cs[i][1]*x + cs[i][2]) + np.random.normal(0, sig10, size=len(x))
    
    return np.clip(f_host, 0, 1)


def k_corr(z, g_minus_r):
    # https://arxiv.org/pdf/1002.2360.pdf
    c = [[0,         0,        0,         0],
         [-0.900332, 3.97338,  0.774394, -1.09389],
         [3.65877,  -8.04213,  11.0321,   0.781176],
         [-16.7457, -31.1241, -17.5553,   0],
         [87.3565,  71.5801,   0,         0],
         [-123.671, 0,         0,         0]]
    
    K = polyval2d(z, g_minus_r, c)
    return K


def g_minus_r_model(M_stellar, mu, cov, seed=None):
    """
    This code is super slow.
    """
    np.random.seed(seed)
    
    x_in = np.log10(M_stellar)
    
    rv = st.multivariate_normal(mean=mu, cov=cov)
    
    # Sample from the PDF
    y = np.linspace(-2, 2, 100) # Range
    
    g_minus_r_draws = np.full(len(x_in), np.nan)
    for i, x_in_i in enumerate(x_in):
        x = np.full_like(y, x_in_i)
        pdf = rv.pdf(x=np.array([x,y]).T)
        g_minus_r_draws[i] = np.random.choice(y, size=1, p=pdf/np.sum(pdf))
            
    # TODO: K-correction
    # This is the galaxy color at ~0.1
    # Now correct it using https://iopscience.iop.org/article/10.1086/510127/pdf
    # to get new g mag
    
    return g_minus_r_draws


def g_minus_r_model_blue(M_stellar, seed=None):    
    mu = np.array([8.9846321,  0.46562084])
    cov = np.array([[0.52527681, 0.06516369],
                   [0.06516369,  0.0229178]])
    return g_minus_r_model(M_stellar, mu, cov, seed=seed)


def g_minus_r_model_red(M_stellar, seed=None):    
    mu = np.array([9.77213478, 0.79589641])
    cov = np.array([[0.23671096, 0.0184602],
                    [0.0184602,  0.00646298]])
    return g_minus_r_model(M_stellar, mu, cov, seed=seed)


def GSMF_blue(M_star, z):
    ratio = _GSMF_blue(M_star)/(_GSMF_red(M_star) + _GSMF_blue(M_star))
    return ratio * GSMF(M_star, z)


def GSMF_red(M_star, z):
    ratio = _GSMF_red(M_star)/(_GSMF_red(M_star) + _GSMF_blue(M_star))
    return ratio * GSMF(M_star, z)


def GSMF_stellar(M_star, z):
    # Currently only defined at z=0
    alpha = 7.45
    beta = 1.05
    M_star_br = 1e11
    M_BH = 10**(alpha + beta*np.log10(M_star/M_star_br))
    return _BHMF_stellar(M_BH)


def _BHMF_stellar(M_BH):
    # Fig. 14 https://arxiv.org/pdf/2110.15607.pdf
    # z=0
    log_M_BH = np.log10(M_BH)
    dat = np.loadtxt('BHMF_stellar.txt', delimiter=',')
    _log_M_BH = dat[:,0]
    _dlog_M_BH = np.diff(log_M_BH)[0]
    _log_phi = dat[:,1]
    # Interpolate
    log_phi = interp1d(_log_M_BH, _log_phi, fill_value='extrapolate')
    return 10**log_phi(log_M_BH)*_dlog_M_BH


def _GSMF_blue(M_star):
    # z=0
    M_br = 10**10.72
    phi = 0.71*1e-3
    alpha = -1.45
    return np.exp(-M_star/M_br)/M_br * phi*(M_star/M_br)**alpha


def _GSMF_red(M_star):
    # z=0
    M_br = 10**10.72
    phi1 = 3.25*1e-3
    phi2 = 0.08*1e-3
    alpha1 = -0.45
    alpha2 = -1.45
    return np.exp(-M_star/M_br)/M_br * (phi1*(M_star/M_br)**alpha1 + phi2*(M_star/M_br)**alpha2)


def GSMF(M_star, z):
    # z<0.1  Use GAMA GSMF https://ui.adsabs.harvard.edu/abs/2012MNRAS.421..621B/abstract
    # Else, use https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.4933A/abstract
    
    # Best-fitting values
    logM_brs = np.array([10.78, 10.71, 10.89, 10.83, 10.86, 10.83, 10.67, 10.60, 10.66, 10.66])
    logphi1s = np.array([np.log10(2.96*1e-3), -2.61, -2.76, -2.60, -2.71, -2.68, -2.75, -2.87, -2.84, -3.19])
    logphi2s = np.array([np.log10(0.63*1e-3), -3.34, -4.12, -3.62, -3.78, -3.63, -3.39, -3.29, -4.39, -4.39])
    alpha1s = np.array([-0.62, -0.47, -1.06, -0.64, -0.80, -0.67, -0.19,  0.00, -0.45, -0.25])
    alpha2s = np.array([-1.50, -1.60, -1.57, -1.61, -1.69, -1.67, -1.58, -1.44, -1.79, -1.94])
    # Uncertainties
    dlogphi1s_0 = 0.434*0.40*1e-3/10**logphi1s[0]
    dlogphi2s_0 = 0.434*0.10*1e-3/10**logphi2s[0]
    dlogM_brs = np.mean([[0.01,0.01], [0.11,0.11], [0.08,0.13], [0.05,0.05], [0.05,0.05],
                         [0.05,0.05], [0.05,0.05], [0.05,0.05], [0.05,0.06], [0.07,0.08]], axis=1)
    dlogphi1s = np.mean([[0.40,0.40], [0.10,0.12], [0.12,0.10], [0.05,0.05],
                         [0.05,0.06], [0.04,0.05], [0.05,0.07], [0.09,0.10], [0.04,0.09], [0.04,0.10]], axis=1)
    dlogphi2s = np.mean([[0.10,0.10], [0.23,0.36], [1.03,3.28], [0.30,0.44],
                         [0.56,1.06], [0.38,0.68], [0.25,0.44], [0.20,0.60], [1.01,2.39], [0.69,2.88]], axis=1)
    dalpha1s = np.mean([[0.03,0.03], [0.44,0.36], [0.59,0.17], [0.22,0.17], [0.30,0.18],
                        [0.28,0.20], [0.35,0.30], [0.37,0.42], [0.40,0.23], [0.54,0.42]], axis=1)
    dalpha2s = np.mean([[0.01,0.01], [0.09,0.12], [0.24,0.65], [0.14,0.18], [0.27,0.47],
                        [0.19,0.32], [0.18,0.28], [0.18,0.47], [0.59,0.78], [0.60,0.70]], axis=1)
    
    z_bins = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    indxs, wt = np.unique(np.digitize(z, z_bins), return_counts=True)
        
    M_br = 10**np.random.normal(loc=logM_brs[indxs], scale=dlogM_brs[indxs])*u.Msun
    phi1 = 10**np.random.normal(loc=logphi1s[indxs], scale=dlogphi1s[indxs])*u.Mpc**-3
    # phi2 (low-mass end) is really hard to constrain well, assume no z evolution
    phi2 = 10**np.random.normal(loc=logphi2s[0], scale=dlogphi2s[0])*u.Mpc**-3
    alpha1 = np.random.normal(loc=alpha1s[0], scale=dalpha1s[0])
    alpha2 = np.random.normal(loc=alpha2s[0], scale=dalpha2s[0])
    
    M_star = np.repeat(np.array([M_star]).T, len(M_br), axis=1)*u.Msun
        
    # We have a phi(M_star,z) at every redshift
    phi = np.exp(-M_star/M_br)/M_br * (phi1*(M_star/M_br)**alpha1 + phi2*(M_star/M_br)**alpha2)
    
    return np.average(phi, axis=1, weights=wt)
        

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


def lambda_A(M_star):
    return 0.1*(np.log10(M_star)/9)**4.5


def ERDF_blue(lambda_Edd, xi=10**-1.65):
    """
    ERDF for blue galaxies (radiatively-efficient, less massive)
    """
    # Lbr = 10**38.1 lambda_br M_BH_br
    # 10^41.67 = 10^38.1 * 10^x * 10^10.66
    lambda_br = 10**np.random.normal(-1.84, np.mean([0.30, 0.37]))
    delta1 = np.random.normal(0.471-0.45, np.mean([0.20, 0.42]))
    delta2 = np.random.normal(2.53, np.mean([0.68, 0.38]))
    # https://ui.adsabs.harvard.edu/abs/2019ApJ...883..139S/abstract
    # What sets the break? Transfer from radiatively efficient to inefficient accretion?
    return xi * ((lambda_Edd/lambda_br)**delta1 + (lambda_Edd/lambda_br)**delta2)**-1 # dN / dlog lambda


def ERDF_red(lambda_Edd, xi=10**-2.13):
    """
    ERDF for red galaxies (radiatively-inefficient, more massive)
    """
    # Lbr = 10**38.1 lambda_br M_BH_br
    # 10^41.67 = 10^38.1 * 10^x * 10^10.66
    lambda_br = 10**np.random.normal(-2.81, np.mean([0.22, 0.14]))
    delta1 = np.random.normal(0.41-0.45, np.mean([0.02, 0.02]))
    delta2 = np.random.normal(1.22, np.mean([0.19, 0.13]))
    # https://ui.adsabs.harvard.edu/abs/2019ApJ...883..139S/abstract
    # What sets the break? Transfer from radiatively efficient to inefficient accretion?
    return xi * ((lambda_Edd/lambda_br)**delta1 + (lambda_Edd/lambda_br)**delta2)**-1 # dN / dlog lambda


def get_RIAF_flux(wav, riaf_sed_path, M_BH=1e6, lambda_Edd=1e-4, z=0.01, s=0.3, p=0.3, alpha=0.3, band='SDSS_g'):
    """
    Compute the SED of a radiatively inefficient accretion flow (RIAF) following Nemmen et al. 2006; Nemmen et al. 2014
    https://academic.oup.com/mnras/article/438/4/2804/2907740
    https://github.com/rsnemmen/riaf-sed
    
    s: power-law index for M(R)
    p: strength of wind (jet vs. AD strength) 
    
    """
    
    bandpass = lib[band]
    # Input redshift
    d_L = cosmo.luminosity_distance(z).to(u.cm)

    R_s = 2*const.G*M_BH*u.Msun/(const.c**2)
    R_s = R_s.to(u.cm)
    R_g = 1/2*R_s
    
    # This equation is not valid, because this disk is truncated
    eta = 0.0572 # for spin = 0
    m9 = M_BH/1e9
    L_bol = (lambda_Edd*1.26*1e38*M_BH)*u.erg/u.s
    dotM = (L_bol/(eta*const.c**2)).to(u.Msun/u.yr)
    dotm = (dotM/(38.8*m9*u.Msun/u.yr)).to(u.dimensionless_unscaled).value
    r_sg = 2150 * m9**(-2/9) * dotm**(4/9) * alpha**(2/9)
    R_sg = r_sg*R_g
    
    Ro = R_sg # outer disk radius = self-gravity radius https://articles.adsabs.harvard.edu/pdf/1989MNRAS.238..897L
    _Ro = Ro/R_s # outer radius in R_s units
        
    # compute the net mass accretion rate into the BH (Yuan et al. 2003)
    #lambda0 = lambda_Edd * (s + 1) * (Ro - R_s) / (Ro - R_s*(R_s/Ro)**s)
    lambdao = lambda_Edd * (R_s/Ro)**(-s)
    dotmo = lambdao # Mass accretion rate at the outer radius R_o of the RIAF
    # should be slightly larger accounting for mass loss from winds
    
    # Convert to fortran format
    dotmo_str = '{:.2e}'.format(dotmo).replace('e', 'd')
    M_BH6_str = '{:.2e}'.format(M_BH/1e6).replace('e', 'd')
    Ro_str = '{:.2e}'.format(_Ro).replace('e', 'd')
    d_pc_str = '{:.2e}'.format(d_L.to(u.pc).value).replace('e', 'd')
    p_str = '{:.2e}'.format(p).replace('e', 'd')
    alpha_str = '{:.2e}'.format(alpha).replace('e', 'd')
    
    print(dotmo_str)
    print(Ro_str)
    
    # Insert the parameters in the input file
    txt = f'# Input parameters for ADAF model\n# ==================================\n#\n# Dynamics\n# ***************************\n# Adiabatic index gamma\ngamai=1.5d0\n# Black hole mass (in 10^6 Solar masses)\nm={M_BH6_str}\n# ratio of gas to total pressure\nbeta=0.9d0\n# alpha viscosity\nalfa={alpha_str}\n# Fraction of turbulent dissipation that directly heats electrons\ndelta=0.2d0\n# Mdot_out (Eddington units)\ndotm0={dotmo_str}\n# R_out (units of R_S)\nrout={Ro_str}\n# p_wind ("strength of wind")\npp0={p_str}\n#\n# Range of eigenvalues of the problem (the "shooting" parameter)\n# Initial and final value, number of models to be computed\nsl0i=1.7\nsl0f=2.5\nnmodels=10\n#\n# Outer boundary conditions (OBCs)\n# T_i (ion temperature) in units of the Virial temperature\nti=0.6\n# T_e (electron temperature) \nte=0.08\n# Mach number=v_R/c_s (radial velocity/sound speed)\nvcs=0.5d0\n#\n# Name of log file\ndiag=out_01\n#\n# SED calculation\n# ***************************\n# distance (in pc)\ndistance={d_pc_str}\n# Inclination angle of outer thin disk (in degrees)\ntheta=30.\n# Spectrum filename\nspec=spec_01\n'
    with open(os.path.join(riaf_sed_path, 'fortran/in.dat'), "w") as text_file:
        text_file.write(txt)
        
    # Execute the SED computation
    cwd = os.getcwd()
    os.chdir(os.path.join(riaf_sed_path, 'fortran'))
    # Find the global solution
    command = os.path.join(riaf_sed_path, 'perl/dyn.pl')
    subprocess.call(command.split())
    # Generate the spectrum
    command = os.path.join(riaf_sed_path, 'perl/spectrum.pl')
    subprocess.call(command.split())
    os.chdir(cwd)
    
    # Open the spectrum file
    dat = np.loadtxt(os.path.join(riaf_sed_path, 'fortran/spec_01'))
    
    nu = 10**dat[:,0]*u.Hz
    wav_sed = np.flip(nu.to(u.nm, equivalencies=u.spectral())*(1 + z))
    
    sed = 10**dat[:,1]*u.erg/u.s/(4*np.pi*d_L**2)
    nuf_nu = np.flip(sed.to(u.erg/u.cm**2/u.s))
    
    # Get L_band
    f_lambda = nuf_nu/wav_sed
    f_lambda_band = bandpass.get_flux(wav_sed, f_lambda)
    f_band = f_lambda_band * bandpass.lpivot
    L_AGN_band = f_band * 4*np.pi*d_L**2
    
    # Get absolute magnitude
    m_band = -2.5*np.log10(f_lambda_band.value) - bandpass.AB_zero_mag
    M_band = m_band - 5*np.log10(d_L.to(u.pc).value) + 5 # This is M_band(z)
    
    fnuf_nu = interp1d(wav_sed.value, nuf_nu.value, kind='slinear', fill_value='extrapolate')
    nuf_nu = fnuf_nu(wav*(1 + z))*nuf_nu.unit
        
    return M_band, m_band, L_AGN_band.to(u.erg/u.s).value, f_band.to(u.erg/u.s/u.cm**2).value, nuf_nu.to(u.erg/u.s/u.cm**2).value


def get_AGN_flux(model_sed, M_BH=1e6, lambda_Edd=0.1, z=0.01, alpha = 0.3, band='SDSS_g'):
    
    bandpass = lib[band]
    # Input redshift
    d_L = cosmo.luminosity_distance(z).to(u.cm)
    d_c = cosmo.comoving_distance(z)
    
    # Self-gravity radius (units of R_g)
    eta = 0.0572 # for spin = 0
    m9 = M_BH/1e9
    L_bol = (lambda_Edd*1.26*1e38*M_BH)*u.erg/u.s
    dotM = (L_bol/(eta*const.c**2)).to(u.Msun/u.yr)
    dotm = (dotM/(38.8*m9*u.Msun/u.yr)).to(u.dimensionless_unscaled).value
    r_sg = 2150 * m9**(-2/9) * dotm**(4/9) * alpha**(2/9)
    log_r_sg = np.clip(np.log10(r_sg), 3, 7) # Falls just slightly below lower bound (2.99451) for IMBHs
    
    # Parameters
    pars = {'bh_mass':M_BH,'dist_c':d_c.to(u.Mpc).value,'lambda_edd':np.log10(lambda_Edd),
            'spin':0,'r_cor':100,'log_r_out':log_r_sg,'kT_e':0.23,'tau':11,'gamma':2.2,'f_pl':0.05,'z':z,'norm':1}
    
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
        
    return M_band, m_band, L_AGN_band.to(u.erg/u.s).value, f_band.to(u.erg/u.s/u.cm**2).value, nuf_nu.to(u.erg/u.s/u.cm**2).value


def lambda_obs(L_bol, seed=None, randomize=True):
    
    np.random.seed(seed)
    L_bol = (L_bol*u.erg/u.s).to(u.Lsun)
    
    if randomize:
        a = np.random.normal(10.96, 0.06, size=len(L_bol))
        b = np.random.normal(11.93, 0.01, size=len(L_bol))
        c = np.random.normal(17.79, 0.10, size=len(L_bol))

        sig = np.random.normal(0.0, 0.2, size=len(L_bol))
    else:
        a = 10.96
        b = 11.93
        c = 17.79
        sig = 0.0
    
    # https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..73D/abstract
    k_X = a*(1 + (np.log10(L_bol/(1*u.Lsun))/b)**c)
    L_X = L_bol/k_X
    L_X = L_X.to(u.erg/u.s)
    
    A = 0.5 # Must be 0.5 so the range is betwen 0 and 1
    l_0 = 43.89 + sig
    sigma_x = 0.46
    
    # https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.3550M/abstract
    # Add 0.2 dex scatter 
    l_x = np.log10(L_X.value)
    lambda_obs = A + 1/np.pi*np.arctan((l_0 - l_x)/sigma_x)
        
    if randomize:
        return np.clip(np.random.normal(lambda_obs, 0.1), 0, 1)
    else:
        return lambda_obs


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
                 D*np.log10(M_BH/1e9)) # + np.random.normal(0, 0.09, size=len(M_BH))) # Delta mag
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
        C = np.random.normal(0.03, 0.04, size=size)
        D = np.random.normal(0.38, 0.05, size=size)
        # Use the pivot mass from Burke 2021
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
               seed_dict={'dc':(lambda x: np.ones_like(x)), 'popIII':f_occ_Bellovary19},
               ERDF_mode=0, log_edd_mu=-1, log_edd_sigma=0.2):

        """
        See https://iopscience.iop.org/article/10.3847/1538-4357/aa803b/pdf

        nbins: Number of stellar mass bins 
        nbootstrap: Number of bootstrap samples (for observational uncertainties)
        eta: Understamping factor (each MC data point corresponds to eta real galaxies)
        zmax:
        ndraw_dim: 
        occ_dict: Dictonary of occupation fractions to use
        ERDF_mode: Which ERDF to adopt
            0 = Weigel 2017
            1 = active fraction
        """
        
        pars = {'nbins':nbins, 'nbootstrap':nbootstrap, 'eta':eta}
        samples = {}
        
        dtype = np.float64
        
        ndraw_dim = int(ndraw_dim)
        pars['ndraw_dim'] = ndraw_dim
        pars['seed_dict'] = seed_dict
        
        pars['log_lambda_min'] = -8.0 # -8
        pars['log_lambda_max'] = 1.0
        
        pars['log_M_star_min'] = 4.5
        pars['log_M_star_max'] = 12
        
        pars['log_M_BH_min'] = 1.5
        pars['log_M_BH_max'] = 9.5
        
        if zmax > 0.25:
            # M_BH - M_star relation from Dong 2020
            print('Using Dong et al. 2020 host-BH mass relation for intermediate redshifts')
            alpha = np.random.normal(loc=0.27, scale=0.08, size=nbootstrap)
            beta = np.random.normal(loc=0.98, scale=0.11, size=nbootstrap)
            M_star_br = 1e10*u.Msun
            M_BH_norm = 1e7*u.Msun
        else:
            # M_BH - M_star relation from Reines 2015
            print('Using Reines et al. 2015 host-BH mass relation for low redshifts')
            alpha = np.random.normal(loc=7.45, scale=0.08, size=nbootstrap)
            beta = np.random.normal(loc=1.05, scale=0.11, size=nbootstrap)
            M_star_br = 1e11*u.Msun
            M_BH_norm = 1*u.Msun

        # Stellar Mass Function
        M_star_ = np.logspace(pars['log_M_star_min'], pars['log_M_star_max'], nbins+1, dtype=dtype)*u.Msun
        dM_star = np.diff(M_star_)
        dlogM_star = np.diff(np.log10(M_star_.value))
        pars['M_star'] = M_star_[1:] + dM_star/2 # bins
        pars['M_star_'] = M_star_

        # 1. Assign number of draws
        d_c_min = 0.5*u.Mpc
        samples['zmax'] = zmax
        samples['zmin'] = z_at_value(cosmo.comoving_distance, d_c_min, zmin=-1e-4, zmax=zmax+1e-4)
        V = cosmo.comoving_volume(zmax)*omega/(4*np.pi)
        pars['V'] = V
        z_samples = np.linspace(samples['zmin'], samples['zmax'])
        dz = np.diff(z_samples)
        z_bins = z_samples[1:] + dz/2

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
        samples['g-r'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=dtype)
        samples['n_i_Edd'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)
        samples['pop'] = np.full([nbootstrap, ndraw_dim], np.nan, dtype=np.int)

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
            
            # 1. Assign redshifts
            # z \propto dV(z)
            z_draw = inv_transform_sampling(cosmo.differential_comoving_volume(z_bins).value, z_samples, n_samples=ndraw_dim)
            
            # 2. Draw from GSMF
            phidM_blue = GSMF_blue(pars['M_star'].value, z_draw)*dM_star
            phidM_red = GSMF_red(pars['M_star'].value, z_draw)*dM_star
            phidM_stellar = GSMF_stellar(pars['M_star'].value, z_draw)*dM_star
            
            # phi = dN / dlog M
            # Normalize
            Vred = V.to(u.Mpc**3).value / eta # Reduced volume
            sf_blue = Vred * trapz((phidM_blue/dM_star).value, pars['M_star'].value)
            sf_red = Vred * trapz((phidM_red/dM_star).value, pars['M_star'].value)
            sf_stellar = Vred * trapz((phidM_stellar/dM_star).value, pars['M_star'].value)
            
            M_star_draw_blue = inv_transform_sampling((phidM_blue/dM_star).value, M_star_.value, survival=sf_blue)*u.Msun
            M_star_draw_red = inv_transform_sampling((phidM_red/dM_star).value, M_star_.value, survival=sf_red)*u.Msun
            M_star_draw_stellar = inv_transform_sampling((phidM_stellar/dM_star).value, M_star_.value, survival=sf_red)*u.Msun
            # Red + blue population
            #M_star_draw = np.concatenate([M_star_draw_blue, M_star_draw_red])
            # Red + blue + stellar population
            M_star_draw = np.concatenate([M_star_draw_stellar, M_star_draw_blue, M_star_draw_red])
            
            #mask_red = np.concatenate([np.full(len(M_star_draw_blue), False), np.full(len(M_star_draw_red), True)])
            
            mask_stellar = np.concatenate([np.full(len(M_star_draw_stellar), True),
                                           np.full(len(M_star_draw_blue), False),
                                           np.full(len(M_star_draw_red), False)])
            mask_red = np.concatenate([np.full(len(M_star_draw_stellar), False),
                                       np.full(len(M_star_draw_blue), False),
                                       np.full(len(M_star_draw_red), True)])
            mask_blue = np.concatenate([np.full(len(M_star_draw_stellar), False),
                                        np.full(len(M_star_draw_blue), True),
                                        np.full(len(M_star_draw_red), False)])
            
            
            #ndraw = len(M_star_draw_blue) + len(M_star_draw_red)
            ndraw = len(M_star_draw_stellar) + len(M_star_draw_blue) + len(M_star_draw_red)
            
            print(np.shape(samples['pop'][j,:ndraw]))
            print(np.shape(samples['pop'][j,:ndraw][mask_stellar]))
            samples['pop'][j,:ndraw][mask_stellar] = np.full_like(mask_stellar, 2)
            samples['pop'][j,:ndraw][mask_red] = 1
            samples['pop'][j,:ndraw][mask_blue] = 0
            
            if ndraw > ndraw_dim:
                print("Warning: ndraw_dim is too small.")
                ndraw = ndraw_dim
                M_star_draw = M_star_draw[:ndraw]
                        
            samples['ndraws'][j] = ndraw
            samples['z_draw'][j,:ndraw] = z_draw[:ndraw]
            samples['M_star_draw'][j,:ndraw] = M_star_draw
            samples['n_i_M'][j,:], _ = np.histogram(M_star_draw, bins=M_star_)

            # Host galaxy colors 
            g_minus_r_draw = np.full(ndraw, np.nan)
            g_minus_r_draw[mask_red] = g_minus_r_model_red(M_star_draw_red.value, seed=j)
            g_minus_r_draw[mask_blue] = g_minus_r_model_blue(M_star_draw_blue.value, seed=j)
            ## Assume blue colors
            g_minus_r_draw[mask_stellar] = g_minus_r_model_blue(M_star_draw_stellar.value, seed=j)
            samples['g-r'][j,:ndraw] = g_minus_r_draw
            
            # 4. BH Mass Function
            M_BH_draw = 10**(alpha[j] + beta[j]*np.log10(M_star_draw/M_star_br) +
                             np.random.normal(0.0, 0.55, size=ndraw))*M_BH_norm
            samples['M_BH_draw'][j,:ndraw] = M_BH_draw
            
            # 5. Eddington ratio Function
            if ERDF_mode == 0:
                lambda_draw = np.full(ndraw, np.nan)
                # Blue  
                xi_blue = ERDF_blue(pars['lambda_Edd']) # dN / dlog lambda
                xi_red = ERDF_red(pars['lambda_Edd']) # dN / dlog lambda
                xi_stellar = ERDF_red(pars['lambda_Edd'])
                norm = trapz(xi_blue + xi_red + xi_stellar, pars['lambda_Edd'])
                xi_blue = xi_blue/norm
                xi_red = xi_red/norm
                ##
                xi_stellar = xi_stellar/norm ##
                lambda_draw[mask_blue] = inv_transform_sampling(xi_blue/dlambda, lambda_, np.count_nonzero(mask_blue))
                lambda_draw[mask_red] = inv_transform_sampling(xi_red/dlambda, lambda_, np.count_nonzero(mask_red))
                ##
                lambda_draw[mask_stellar] = inv_transform_sampling(xi_stellar/dlambda, lambda_, np.count_nonzero(mask_stellar))
            elif ERDF_mode == 1:
                p = lambda_A(M_star_draw.value)
                lambda_draw_init = 10**np.random.normal(log_edd_mu, log_edd_sigma, ndraw)
                lambda_draw = survival_sampling(lambda_draw_init, survival=p, fill_value=1e-8)

            samples['lambda_draw'][j,:ndraw] = lambda_draw
            samples['n_i_Edd'][j,:], _ = np.histogram(lambda_draw, bins=lambda_)
            
            # Occupation probabililty survival sampling
            # Don't include stellar mass "galaxies"
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
        ndraw_dim = int(np.max(samples['ndraws']))
        for key in samples.keys():
            if key.startswith('n_i'):
                samples[key] = samples[key].astype(np.float)*eta
            elif '_draw' in key:
                # Free up some memory
                samples[key] = samples[key][:ndraw_dim]
        
        self.pars = pars
        self.samples = samples
        
        
    def sample_sed_grid(self, w0=1e-3, w1=1e8, band='SDSS_g', model_sed_name='optxagnf', nbins=8, save_fits=False, load_fits=False,
                       sed_pars={'bh_mass':1e8,'dist_c':30.0,'lambda_edd':np.log10(0.1),'spin':0,'r_cor':100,
                                 'log_r_out':-1,'kT_e':0.23,'tau':11,'gamma':2.2,'f_pl':0.05,'z':0.007,'norm':1}):
        
        pars = self.pars
        s = self.samples
                
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
        x = np.logspace(2, 9, nbins) # hard-code this because the SEDs are not well-defined outside this range
        y = np.logspace(pars['log_lambda_min'], pars['log_lambda_max'], nbins)
        z = np.linspace(s['zmin'], s['zmax'], nbins)
        
        print(f'Creating SED grid in band {band}')
        
        if load_fits:
            pass
        else:
            # Get AGN luminosity in band and M_i(z=2)
            vget_AGN_flux = np.vectorize(get_AGN_flux, otypes=[np.float,np.float,np.float,np.float,np.ndarray])

            X, Y, Z = np.meshgrid(x, y, z, indexing='ij', sparse=True)
            _, _, L_band_model, _, nuf_nu = vget_AGN_flux(model_sed, M_BH=X, lambda_Edd=Y, z=Z, band=band)

            X, Y = np.meshgrid(x, y, indexing='ij', sparse=True)
            M_i_model, _, _, _, _ = vget_AGN_flux(model_sed, M_BH=X, lambda_Edd=Y, z=2.0, band='SDSS_i')
                
        self.samples[f'L_{band}_model'] = L_band_model*u.erg/u.s # Shape NxNxN
        self.samples['M_i_model'] = M_i_model # Shape NxNxN
                
        # Create interpolator objects
        fn_L_band_model = RegularGridInterpolator((np.log10(x), np.log10(y), z), np.log10(L_band_model),
                                                  bounds_error=False, fill_value=None)
        fn_M_i_model = RegularGridInterpolator((np.log10(x), np.log10(y)), M_i_model,
                                               bounds_error=False, fill_value=None)
                
        for k, seed in enumerate(pars['seed_dict']):
            
            # MUST initialize L_AGN to 0, so host-galaxy mag is finite even when unoccupied!
            self.samples[f'L_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], 0.0)*u.erg/u.s
            self.samples[f'M_i_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan)
            
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
                
                # Obscured fraction
                p = 1 - lambda_obs(self.samples[f'L_draw_{seed}'].value[j,:ndraw][mask_occ], seed=j)
                L_band = self.samples[f'L_{band}_{seed}'][j,:ndraw][mask_occ]
                L_band_obs = survival_sampling(L_band.value, survival=p, fill_value=0.0)*u.erg/u.s
                self.samples[f'L_{band}_{seed}'][j,:ndraw][mask_occ] = L_band_obs
                self.samples[f'M_i_{seed}'][j,:ndraw][mask_occ][L_band_obs.value==0] = np.nan
                
        
        if save_fits:
            energies = model_sed.energies(0)[::-1]*u.keV # Why N-1?
            wav = (energies[:-1]).to(u.nm, equivalencies=u.spectral()) # RF

            c1 = fits.Column(name='log_M_BH', array=np.log10(x), format='D')
            c2 = fits.Column(name='log_LAMBDA_EDD', array=np.log10(y), format='D')
            c3 = fits.Column(name='Z', array=z, format='D')
            table_hdu0 = fits.BinTableHDU.from_columns([c1, c2, c3])

            c1 = fits.Column(name='log_WAV', array=np.log10(wav.value), format='D')
            table_hdu1 = fits.BinTableHDU.from_columns([c1])

            # Create FITS file
            nuf_nu = np.array(nuf_nu.tolist())
            hdu0 = fits.PrimaryHDU(nuf_nu)

            table_hdu = fits.HDUList([hdu0, table_hdu0, table_hdu1])
            table_hdu.writeto('sed_grid.fits', overwrite=True)
            # M_i can be obtained using the approximation M_i = 125 - 3.3 log(L_bol / erg s^−1)

        return
    
    
    def sample_light_curves(self, t_obs, dt_min=10, band='SDSS_g', SFinf_small=10**-2.5, m_5=25.0):
        
        pars = self.pars
        s = self.samples
        
        ndraw_dim = int(np.max(s['ndraws']))
        nbootstrap = pars['nbootstrap']
        
        t_obs_dense = np.arange(np.min(t_obs), np.max(t_obs), dt_min)
        t_rest_dense = np.arange(np.min(t_obs), np.max(t_obs), dt_min)
        
        s['lc_t_obs'] = t_obs
                        
        for k, seed in enumerate(pars['seed_dict']):
            
            print(f'Sampling light curves with seeding mechanism {seed}')
            
            s[f'lc_{band}_{seed}_idx'] = np.full([nbootstrap, ndraw_dim], np.arange(ndraw_dim), dtype=np.float64)
            s[f'm_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan)
            s[f'm_host_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan)
            s[f'SFinf_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan)
            s[f'tau_RF_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan)
            s[f'L_host_{band}_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan)*u.erg/u.s
            s[f'g_minus_r_{seed}'] = np.full([nbootstrap, ndraw_dim], np.nan)

            for j in tqdm(range(nbootstrap)):

                ndraw = int(s['ndraws'][j])

                z = s['z_draw'][j,:ndraw]
                M_BH = s[f'M_BH_draw_{seed}'][j,:ndraw].value
                M_star = s['M_star_draw'][j,:ndraw]
                g_minus_r = s['g-r'][j,:ndraw]
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
                f_host = f_host_model(z, M_star.value, seed=j)
                
                # This is r or g-band luminosity from the M/L ratio
                L_band_host = f_host * (M_star/(1*u.Msun) / 10**(b*g_minus_r + a + color_var))*u.Lsun
                L_band_host = L_band_host.to(u.erg/u.s)
                
                d_L = cosmo.comoving_distance(z).to(u.cm)
                
                # Get apparent magnitude of AGN + host galaxy
                if band == 'GROUND_COUSINS_R':
                    # r -> R color correction
                    band_r = 'SDSS_r'
                    
                    # These are r-band fluxes from the M/L ratio
                    f_band_AGN = L_band_AGN / (4*np.pi*d_L**2) # (R)
                    f_band_host = L_band_host / (4*np.pi*d_L**2) # (r)
                    f_lambda_band_AGN = (f_band_AGN / lib[band].lpivot).to(u.erg/u.s/u.cm**2/u.AA) # (R)
                    f_lambda_band_host = (f_band_host / lib[band_r].lpivot).to(u.erg/u.s/u.cm**2/u.AA) # (r)
                    
                    # r-band magnitudes
                    m_band_AGN = -2.5*np.log10(f_lambda_band_AGN.value) - lib[band].AB_zero_mag # (R)
                    r_band_host = -2.5*np.log10(f_lambda_band_host.value) - lib[band_r].AB_zero_mag # (r)
                    
                    # http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php#Lupton2005
                    m_band_host = r_band_host - 0.1837*g_minus_r - 0.0971 # (R)
                    
                    # Magnitude addition formula AGN + host
                    #m_band = -2.5*np.log10(10**(-0.4*m_band_host) + 10**(-0.4*m_band_AGN)) # (R)
                    ## methinks AGN mag is too bright
                    
                    # Convert back to luminosity in R
                    f_lambda_band_host = 10**(-0.4*m_band_host) * lib[band].AB_zero_flux # (R)
                    f_band_host = (f_lambda_band_host * lib[band].lpivot).to(u.erg/u.s/u.cm**2) # (R)
                    L_band_host = f_band_host * (4*np.pi*d_L**2) # (R)
                    L_band_host = L_band_host.to(u.erg/u.s)
                    
                # Host-galaxy K-correction
                if band == 'SDSS_g':
                    f_band_host = L_band_host / (4*np.pi*d_L**2)
                    f_lambda_band_host = (f_band_host / lib[band].lpivot).to(u.erg/u.s/u.cm**2/u.AA)
                    # K-correction
                    print(k_corr(z, g_minus_r))
                    m_band_host = -2.5*np.log10(f_lambda_band_host.value) - lib[band].AB_zero_mag + k_corr(z, g_minus_r)
                    # Convert back to luminosity
                    f_lambda_band_host = 10**(-0.4*m_band_host) * lib[band].AB_zero_flux
                    f_band_host = (f_lambda_band_host * lib[band].lpivot).to(u.erg/u.s/u.cm**2)
                    #L_band_host = f_band_host * (4*np.pi*d_L**2)
                    #L_band_host = L_band_host.to(u.erg/u.s) # (g)
                    
                # No color correction
                f_band = (L_band_host + L_band_AGN) / (4*np.pi*d_L**2)
                #f_band_host = L_band_host / (4*np.pi*d_L**2)

                f_lambda_band = (f_band / lib[band].lpivot).to(u.erg/u.s/u.cm**2/u.AA)
                #f_lambda_band_host = (f_band_host / lib[band].lpivot).to(u.erg/u.s/u.cm**2/u.AA)

                m_band = -2.5*np.log10(f_lambda_band.value) - lib[band].AB_zero_mag
                #m_band_host = -2.5*np.log10(f_lambda_band_host.value) - lib[band].AB_zero_mag
                    
                # Fluxes
                s[f'm_{band}_{seed}'][j,:ndraw] = m_band
                s[f'L_host_{band}_{seed}'][j,:ndraw] = L_band_host
                                
                # Draw SF_\infty and tau (rest-frame)
                # In M10, M_i is basically a proxy for L_bol, so we need to use the Shen relation
                # even for IMBHs, to preserve the linearity in the extrapolation of this relation
                M_i_AGN = 90 - 2.5*np.log10(L_bol_AGN.to(u.erg/u.s).value)
                
                SFinf = draw_SFinf(lambda_RF.to(u.AA).value, M_i_AGN, M_BH, size=ndraw)
                tau = draw_tau(lambda_RF.to(u.AA).value, M_i_AGN, M_BH, size=ndraw)
                
                # Host-galaxy dilution
                dL = L_band_AGN*np.log(10)/2.5*SFinf #* 1.3 # The 1.3 is a fudge factor to normalize with the qsos
                SFinf = 2.5/np.log(10)*dL/(L_band_AGN + L_band_host)

                mask_small = (SFinf < SFinf_small) | (M_BH < 1e2) | (m_band > m_5+1)
                
                shape = np.count_nonzero(~mask_small)
                t_obs_dense_shaped = (np.array([t_obs_dense]*shape)).T
                t_rest_dense_shaped = (np.array([t_rest_dense]*shape)).T / (1 + z[~mask_small])

                s[f'SFinf_{band}_{seed}'][j,:ndraw] = SFinf
                s[f'tau_RF_{band}_{seed}'][j,:ndraw] = tau
                
                # Simulate light curves (clip tau to avoid numerical issues if tau << dt_min)
                # TODO: try dt_min instead of 2 dt_min?
                lcs = simulate_drw(t_rest_dense_shaped, np.clip(tau, 2*dt_min, None)[~mask_small],
                                   z[~mask_small], m_band[~mask_small], SFinf[~mask_small]).T
                f = interp1d(t_obs_dense, lcs, fill_value='extrapolate')
                s[f'lc_{band}_{seed}_{j}'] = f(t_obs)
                s[f'lc_{band}_{seed}_idx'][j,ndraw:] = np.nan
                s[f'lc_{band}_{seed}_idx'][j,:ndraw][mask_small] = np.nan
                
        return
                            

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
            
        
    def plot(self, figsize=(3*6*0.9, 2*5*0.9), seed_colors=['b','m'], seed_markers=['s','o'], moct=np.nanmean, n_bin_min=10):
    
        import matplotlib.ticker as ticker

        fig, axs = plt.subplots(2, 3, figsize=figsize)
        axs = axs.flatten()
        
        pars = self.pars
        samples = self.samples
        
        ndraw_dim = int(np.max(samples['ndraws']))
        
        V = pars['V']
        n_i_M = samples['n_i_M'][:,:ndraw_dim]
        n_i_Edd = samples['n_i_Edd'][:,:ndraw_dim]
        
        # Clean
        n_i_M[:,n_i_M[0] < n_bin_min] = np.nan
        n_i_Edd[:,n_i_Edd[0] < n_bin_min] = np.nan
        
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
        axs[0].scatter(M_star, moct(n_i_M, axis=0)/dlogM_star/V, lw=3, color='k')

        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel(r'$M_{\star}\ (M_{\odot})$', fontsize=18)
        axs[0].set_ylabel(r'$\phi(M_{\star})$ (dex$^{-1}$ Mpc$^{-3}$)', fontsize=18)
        axs[0].set_xlim([1e5, 4e11])
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
            axs[1].scatter(M_star, moct(f, axis=0), lw=3, color=seed_colors[k], marker=seed_markers[k])
        
        axs[1].set_xlabel(r'$M_{\star}\ (M_{\odot})$', fontsize=18)
        axs[1].set_ylabel(r'$\lambda_{\rm{occ}}$', fontsize=18)
        axs[1].set_xscale('log')
        axs[1].set_xlim([1e5, 4e11])
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
            axs[3].scatter(M_BH, moct(samples[f'n_i_M_{seed}'][:,:ndraw_dim], axis=0)/dlogM_BH/V,
                           lw=3, color=seed_colors[k], marker=seed_markers[k])

        axs[3].set_xlabel(r'$M_{\rm{BH}}\ (M_{\odot})$', fontsize=18)
        axs[3].set_ylabel(r'$\phi(M_{\rm{BH}})$ (dex$^{-1}$ Mpc$^{-3}$)', fontsize=18)
        axs[3].set_xscale('log')
        axs[3].set_yscale('log')
        axs[3].set_xlim([1e2, 1e8])
        axs[3].set_ylim([1e-4, 1e0])

        # 5. Eddington ratio distribution (ERDF) function
        norm = np.nansum(moct(n_i_Edd, axis=0)*dloglambda)
        axs[4].scatter(lambda_Edd, moct(n_i_Edd, axis=0)/dloglambda/norm, lw=3, color='k')
        axs[4].fill_between(lambda_Edd, np.nanpercentile(n_i_Edd, 16, axis=0)/dloglambda/norm,
                            np.nanpercentile(n_i_Edd, 84, axis=0)/dloglambda/norm, color='k', alpha=0.5)

        axs[4].set_xlabel(r'$\lambda_{\rm{Edd}}$', fontsize=18)
        axs[4].set_ylabel(r'$\xi$ (dex$^{-1}$)', fontsize=18)
        axs[4].set_xscale('log')
        axs[4].set_yscale('log')
        axs[4].set_xlim([5e-8, 1e1])
        axs[4].set_ylim([1e-4, 1e1])
        
        # 6. AGN Luminosity Function
        for k, seed in enumerate(pars['seed_dict']):
            
            n_i_L = samples[f'n_i_L_{seed}'][:,:ndraw_dim]
            n_i_L[:,n_i_L[0] < n_bin_min] = np.nan
            
            mask_L = L.value < 1e45
            
            axs[5].scatter(L[mask_L], moct(n_i_L, axis=0)[mask_L]/dlogL/V,
                           lw=3, color=seed_colors[k], marker=seed_markers[k])
            axs[5].fill_between(L[mask_L], np.nanpercentile(n_i_L, 16, axis=0)[mask_L]/dlogL/V,
                            np.nanpercentile(n_i_L, 84, axis=0)[mask_L]/dlogL/V,
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
        axs[5].set_ylim([1e-9, None])

        import string, matplotlib
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

            locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
            ax.xaxis.set_major_locator(locmaj)
            locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            
            if i != 1:
                locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
                ax.yaxis.set_major_locator(locmaj)
                locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
                ax.yaxis.set_minor_locator(locmin)
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


        fig.tight_layout()
        return fig