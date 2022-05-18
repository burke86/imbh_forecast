import sys, os
import pickle

from pathlib import Path
import re
import subprocess
import pickle

import numpy as np
from astropy import units as u
import astropy.constants as const
import scipy.stats as st
from scipy.interpolate import interp1d, RegularGridInterpolator, LinearNDInterpolator
from numpy.polynomial.polynomial import polyval2d
from scipy.integrate import trapz, cumtrapz
from astropy.io import fits
from fast_histogram import histogram1d
from qsofit import qso_fit

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

def pm_prec(mag, gamma=0.038, m_5=25.0, sigma_sys=0.003):
    """
    Model for photometric precision of a survey following
    https://ui.adsabs.harvard.edu/abs/2019ApJ...873..111I/abstract
    """
    x = 10**(0.4*(mag - m_5))
    sigma_rand = np.sqrt((0.04 - gamma)*x + gamma*x**2)
    return np.sqrt(sigma_sys**2 + sigma_rand**2)

def hist1d(x, bins):
    """
    Wrapper for fast 1d histograms to mimic the numpy.histogram behavoir
    """
    log_bins = np.log10(bins)
    log_x = np.log10(x[x>0])
    #return np.histogram(x, bins)
    h = histogram1d(log_x, bins=len(bins)-1, range=[np.min(log_bins), np.max(log_bins)])
    return h, bins

def f_host_model(z, M_star, seed=None):
    """
    Return aperture covering factor for a host galaxy given its redshift and stellar mass.
    Model constructed from fitting fstar vs. Mstar in bins of log Mstar
    """
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
    """
    Approximate k-correction for host galaxies given redshift and g-r color
    # https://arxiv.org/pdf/1002.2360.pdf
    """
    c = [[0,         0,        0,         0],
         [-0.900332, 3.97338,  0.774394, -1.09389],
         [3.65877,  -8.04213,  11.0321,   0.781176],
         [-16.7457, -31.1241, -17.5553,   0],
         [87.3565,  71.5801,   0,         0],
         [-123.671, 0,         0,         0]]
    
    K = polyval2d(z, g_minus_r, c)
    return K


def g_minus_r_model(M_star, mu, cov, seed=None, len_chunk=None):
    """
    Sample host galaxy colors given stellar mass and 2D Gaussian PDF.
    Creates a grid of PDFs in stellar mass bins as an efficient approximation
    https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
    https://peterroelants.github.io/posts/multivariate-normal-primer/
    """    
    rv = st.multivariate_normal(mean=mu, cov=cov)
    A = cov[0,0]
    B = cov[1,1]
    C = cov[0,1]
        
    # Sample from the PDF
    y = np.linspace(-2, 2, 50) # Range for colors
    x_in_grid_ = np.linspace(4, 12, 20) # Range for stellar mass
    dx = np.diff(x_in_grid_)
    x_in_grid = x_in_grid_[1:] + dx/2 # bin centers
    
    # Digitize log M_star
    idx_bin = np.searchsorted(x_in_grid_[1:], np.log10(M_star), side='left')
        
    # Create grid of PDFs instead of evaluating at each M_star to speed-up
    g_minus_r_draws = np.empty(len(M_star))
    
    cov_ygivenx = B - (C * (1/A) * C)
    mus_ygivenx = mu[1] + (C * (1/A) * (x_in_grid - mu[0]))
    for i, x_i in enumerate(x_in_grid):
        # Conditional probability distribution function p(y|x=M_star)
        rv = st.norm(loc=mus_ygivenx[i], scale=np.sqrt(cov_ygivenx))
        mask = (idx_bin==i)
        g_minus_r_draws[mask] = rv.rvs(size=np.count_nonzero(mask))
        
    return g_minus_r_draws
        
    # Use instead https://peterroelants.github.io/posts/multivariate-normal-primer/
    

def g_minus_r_model_blue(M_stellar, seed=None):
    """
    Draw g-r host galaxy colors for the blue/green galaxy population given stellar mass
    """
    mu = np.array([8.9846321,  0.46562084])
    cov = np.array([[0.52527681, 0.06516369],
                   [0.06516369,  0.0229178]])
    return g_minus_r_model(M_stellar, mu, cov, seed=seed)


def g_minus_r_model_red(M_stellar, seed=None):
    """
    Draw g-r host galaxy colors for the red galaxy population given stellar mass
    """
    mu = np.array([9.77213478, 0.79589641])
    cov = np.array([[0.23671096, 0.0184602],
                    [0.0184602,  0.00646298]])
    return g_minus_r_model(M_stellar, mu, cov, seed=seed)


def GSMF_blue(M_star, z):
    """
    Redshift-dependent galaxy stellar mass function of the blue+green galaxy population (single Schechter function)
    assuming the ratio of the blue/red GSMF doesn't change with time (probably wrong)
    """
    ratio = _GSMF_blue(M_star)/(_GSMF_red(M_star) + _GSMF_blue(M_star))
    return ratio * GSMF(M_star, z)


def GSMF_red(M_star, z):
    """
    Redshift-dependent galaxy stellar mass function of the red galaxy population (single Schechter function)
    assuming the ratio of the blue/red GSMF doesn't change with time (probably wrong)
    """
    ratio = _GSMF_red(M_star)/(_GSMF_red(M_star) + _GSMF_blue(M_star))
    return ratio * GSMF(M_star, z)

def BHMF_wandering(M_BH):
    """
    Black hole mass function (BHMF) anchored to LIGO/VIRGO GW merger rates below 10^2 Msun
    Fig. 14 of https://ui.adsabs.harvard.edu/abs/2022ApJ...924...56S/abstract
    Low-mass end of the Schechter function is anchored to 10^4 Msun
    """
    phi_ = 10**np.random.normal(-1, 0.3)
    M_BH_br = 1e4
    alpha = -1.6
    phidM = phi_*(M_BH/M_BH_br)**alpha
    return phidM

def _GSMF_blue(M_star):
    """
    Galaxy stellar mass function of the z ~ 0 blue+green galaxy population (single Schechter function)
    https://ui.adsabs.harvard.edu/abs/2012MNRAS.421..621B/abstract
    """
    M_br = 10**10.72
    phi = 0.71*1e-3
    alpha = -1.45
    phi = np.exp(-M_star/M_br)/M_br * phi*(M_star/M_br)**alpha
    return phi


def _GSMF_red(M_star):
    """
    Galaxy stellar mass function of the z ~ 0 red galaxy population (double Schechter function)
    https://ui.adsabs.harvard.edu/abs/2012MNRAS.421..621B/abstract
    """
    M_br = 10**10.72
    phi1 = 3.25*1e-3
    phi2 = 0.08*1e-3
    alpha1 = -0.45
    alpha2 = -1.45
    phi = np.exp(-M_star/M_br)/M_br * (phi1*(M_star/M_br)**alpha1 + phi2*(M_star/M_br)**alpha2)
    return phi


def GSMF(M_star, z, seed=None):
    """
    Galaxy stellar mass function with redshift evolution (double Schechter function)
    https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.4933A/abstract
    If z < 0.1, use GAMA GSMF: https://ui.adsabs.harvard.edu/abs/2012MNRAS.421..621B/abstract
    which is better-constrained in the dwarf galaxy regime
    """
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


def f_occ_Bellovary19(M_star):
    """
    Heavy seed scenario
    """
    f = np.array([0.03, 0.06, 0.16, 0.2, 0.78, 1.0, 1.0, 1.0, 1.0])
    x = np.array([4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
    f_interp = np.interp(np.log10(M_star), x, f)
    df_interp = np.interp(np.log10(M_star), x, 0.3*f)
    return np.clip(f_interp, 0, 1)


def f_occ_heavyMS(M_star):
    """
    Heavy-MS seed scenario
    """
    f = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.4, 0.67, 0.95, 1.0, 1.0 , 1.0 , 1.0])
    x = np.array([4.5, 5.5, 6.5, 7.5, 7.9, 8.3, 8.85, 9.35, 9.5, 10.5, 11.5, 12.5])
    f_interp = np.interp(np.log10(M_star), x, f)
    return np.clip(f_interp, 0, 1)

def lambda_A(M_star):
    return 0.1*(np.log10(M_star)/9)**4.5


def ERDF_blue(lambda_Edd, xi=10**-1.65, seed=None):
    """
    ERDF for blue galaxies (radiatively-efficient, less massive)
    """
    # Lbr = 10**38.1 lambda_br M_BH_br
    # 10^41.67 = 10^38.1 * 10^x * 10^10.66
    lambda_br = 10**np.random.normal(-1.84, np.mean([0.30, 0.37]))
    delta1 = np.random.normal(0.471-0.7, np.mean([0.02, 0.02])) # -0.7 #np.random.normal(0.471, np.mean([0.20, 0.42])) # -0.45
    delta2 = np.random.normal(2.53, np.mean([0.68, 0.38]))
    # https://ui.adsabs.harvard.edu/abs/2019ApJ...883..139S/abstract
    # What sets the break? Transfer from radiatively efficient to inefficient accretion?
    return xi * ((lambda_Edd/lambda_br)**delta1 + (lambda_Edd/lambda_br)**delta2)**-1 # dN / dlog lambda


def ERDF_red(lambda_Edd, xi=10**-2.13, seed=None):
    """
    ERDF for red galaxies (radiatively-inefficient, more massive)
    """
    # Lbr = 10**38.1 lambda_br M_BH_br
    # 10^41.67 = 10^38.1 * 10^x * 10^10.66
    lambda_br = 10**np.random.normal(-2.81, np.mean([0.22, 0.14]))
    delta1 = np.random.normal(0.41-0.7, np.mean([0.02, 0.02])) # This is not sensitive
    delta2 = np.random.normal(1.22, np.mean([0.19, 0.13]))
    # https://ui.adsabs.harvard.edu/abs/2019ApJ...883..139S/abstract
    # What sets the break? Transfer from radiatively efficient to inefficient accretion?
    return xi * ((lambda_Edd/lambda_br)**delta1 + (lambda_Edd/lambda_br)**delta2)**-1 # dN / dlog lambda


def get_RIAF_flux(model_sed, M_BH=1e6, lambda_Edd=1e-4, z=0.01, M_BH_template=4e6, lambda_Edd_template=1e-4, z_template=0.01,
                  s=0.3, p=2.3, alpha=0.3, beta=0.9, delta=0.001, gamma=1.5, theta=30, sl0i=2.0, sl0f=3.0, band='SDSS_g'):
    """
    Compute the SED of a radiatively inefficient accretion flow (RIAF) following Nemmen et al. 2006; Nemmen et al. 2014
    https://academic.oup.com/mnras/article/438/4/2804/2907740
    https://github.com/rsnemmen/riaf-sed
    
    The RIAF SED code doesn't really work, so just use some template parameters as a template and re-scale everything after
    like L ~ lambda_Edd and L ~ M_BH
    
    To fit a SED, delta and p make the biggest differences
    
    s: power-law index for M(R)
    p: strength of wind (jet vs. AD strength)
    alpha: SSD viscosity parameter [do not recommend changing]
    beta: ratio of gas to total pressure [do not recommend changing]
    delta: fraction of turbulent dissipation that directly heats electrons
    gamma: adiabatic index [do not recommend changing]
    theta: inclination angle [deg]
    """
        
    # Vectorization
    wav, riaf_sed_path = model_sed['wav'], model_sed['dir']
    
    # Normalization
    lambda_Edd = lambda_Edd * 5e6
    
    bandpass = lib[band]
    # Input redshift
    d_L = cosmo.luminosity_distance(z).to(u.cm)
    d_L_template = cosmo.luminosity_distance(z_template).to(u.cm)

    R_s = 2*const.G*M_BH*u.Msun/(const.c**2)
    R_s = R_s.to(u.cm)
    R_g = 1/2*R_s
    
    # This equation is not valid, because this disk is truncated
    eta = 0.0572 # for spin = 0
    m9 = M_BH_template/1e9
    L_bol = (lambda_Edd_template*1.26*1e38*M_BH_template)*u.erg/u.s
    dotM = (L_bol/(eta*const.c**2)).to(u.Msun/u.yr)
    dotm = (dotM/(38.8*m9*u.Msun/u.yr)).to(u.dimensionless_unscaled).value
    r_sg = 2150 * m9**(-2/9) * dotm**(4/9) * alpha**(2/9)
    R_sg = r_sg*R_g
    
    dotM_Edd = 22*m9*u.Msun/u.yr # 10% radiative efficiency
    
    Ro = R_sg # outer disk radius = self-gravity radius https://articles.adsabs.harvard.edu/pdf/1989MNRAS.238..897L
    _Ro = Ro/R_s # outer radius in R_s units
        
    # compute the net mass accretion rate into the BH (Yuan et al. 2003)
    # should be slightly larger accounting for mass loss from winds
    dotMo = dotM * (R_s/Ro)**(-s)
    dotmo = dotMo/dotM_Edd # Mass accretion rate at the outer radius R_o of the RIAF
    
    to_fortran = lambda x: '{:.2e}'.format(x).replace('e', 'd')
    
    # Convert to fortran format
    dotmo_str = to_fortran(dotmo)
    M_BH6_str = to_fortran(M_BH_template/1e6)
    Ro_str = to_fortran(_Ro)
    d_pc_str = to_fortran(d_L_template.to(u.pc).value)
    p_str = to_fortran(p)
    alpha_str = to_fortran(alpha)
    beta_str = to_fortran(beta)
    delta_str = to_fortran(delta)
    gamma_str = to_fortran(gamma)
    theta_str = to_fortran(theta)
    
    sl0i_str = to_fortran(sl0i)
    sl0f_str = to_fortran(sl0f)
    
    #print(dotmo_str)
    
    # Insert the parameters in the input file
    txt = f'# Input parameters for ADAF model\n# ==================================\n#\n# Dynamics\n# ***************************\n# Adiabatic index gamma\ngamai={gamma_str}\n# Black hole mass (in 10^6 Solar masses)\nm={M_BH6_str}\n# ratio of gas to total pressure\nbeta={beta_str}\n# alpha viscosity\nalfa={alpha_str}\n# Fraction of turbulent dissipation that directly heats electrons\ndelta={delta_str}\n# Mdot_out (Eddington units)\ndotm0={dotmo_str}\n# R_out (units of R_S)\nrout={Ro_str}\n# p_wind ("strength of wind")\npp0={p_str}\n#\n# Range of eigenvalues of the problem (the "shooting" parameter)\n# Initial and final value, number of models to be computed\nsl0i={sl0i_str}\nsl0f={sl0f_str}\nnmodels=10\n#\n# Outer boundary conditions (OBCs)\n# T_i (ion temperature) in units of the Virial temperature\nti=0.6\n# T_e (electron temperature) \nte=0.08\n# Mach number=v_R/c_s (radial velocity/sound speed)\nvcs=0.9d0\n#\n# Name of log file\ndiag=out_{M_BH6_str}_{dotmo_str}\n#\n# SED calculation\n# ***************************\n# distance (in pc)\ndistance={d_pc_str}\n# Inclination angle of outer thin disk (in degrees)\ntheta={theta_str}\n# Spectrum filename\nspec=spec_{M_BH6_str}_{dotmo_str}\n'
    
    out_filename = f'fortran/out_{M_BH6_str}_{dotmo_str}'
    spec_filename = f'fortran/spec_{M_BH6_str}_{dotmo_str}'
    
    run = True
    # If the input file exists already and has the same parameters, skip running
    if os.path.exists(os.path.join(riaf_sed_path, 'fortran/in.dat')):
        with open(os.path.join(riaf_sed_path, 'fortran/in.dat'), 'r') as text_file:
            txt_old = text_file.read()
            if txt_old == txt:
                with open(os.path.join(riaf_sed_path, out_filename), 'r') as out_file:
                    dat_out = out_file.read()
                    if 'Run finished' in dat_out:
                        #print('Re-using previous run.')
                        run = False
                    
    if run:
        with open(os.path.join(riaf_sed_path, 'fortran/in.dat'), 'w') as text_file:
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
    
    # Make sure the model ran succesfully
    with open(os.path.join(riaf_sed_path, out_filename), 'r') as out_file:
        dat_out = out_file.read()
        if not 'Run finished' in dat_out:
            raise ValueError('RIAF-SED failed to find a solution.')
    
    # Open the spectrum file
    dat = np.loadtxt(os.path.join(riaf_sed_path, spec_filename))
    
    nu = 10**dat[:,0]*u.Hz
    wav_sed = np.flip(nu.to(u.nm, equivalencies=u.spectral())*(1 + z))
    
    # Disk temperature correction (this is almost certaintly wrong)
    # See Done et al. 2012
    T_M = (M_BH/M_BH_template)**(-1/4)
    T_l = (lambda_Edd/lambda_Edd_template)**(1/4)
    # Wein's law lambda ~ 1/T
    wav_sed = wav_sed/T_M
    wav_sed = wav_sed/T_l
    
    # K-correction
    wav_sed = wav_sed/(1 + z_template)*(1 + z)
    
    sed = 10**dat[:,1]*u.erg/u.s/(4*np.pi*d_L**2)
    nuf_nu = np.flip(sed.to(u.erg/u.cm**2/u.s))
    
    # Re-scale SED
    nuf_nu = nuf_nu*(lambda_Edd/lambda_Edd_template)
    nuf_nu = nuf_nu*(M_BH/M_BH_template)
    
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


def get_AGN_flux(model_sed, M_BH=1e6, lambda_Edd=0.1, z=0.01, spin=0, r_corr=100, alpha=0.3, kT_e=0.23, tau=11, gamma=2.2, f_pl=0.05, band='SDSS_g'):
    """
    Compute flux from AGN SED using the Done+12 xspec model
    """
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
            'spin':spin,'r_cor':r_corr,'log_r_out':log_r_sg,'kT_e':kT_e,'tau':tau,
            'gamma':gamma,'f_pl':f_pl,'z':z,'norm':1}
    
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
    """
    Compute the luminosity-dependent optically-obscured AGN fraction at L_bol
    """
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
    """
    Generate damped random walk on grid
    """
    for i in range(1, N):
        dt = t_obs[i,:] - t_obs[i - 1,:]
        x[i,:] = (x[i - 1,:] - dt * (x[i - 1,:] - xmean) + np.sqrt(2) * SFinf * E[i,:] * np.sqrt(dt))
    return x


def simulate_drw(t_rest, tau=300., z=2.0, xmean=0, SFinf=0.3):
    """
    Simulate damped random walk on grid given tau, SF
    """
    N = np.shape(t_rest)[0]
    ndraw = len(tau)
    
    # t_rest [N, ndraw]

    t_obs = t_rest * (1. + z) / tau
    
    x = np.zeros((N, ndraw))
    x[0,:] = np.random.normal(xmean, SFinf)
    E = np.random.normal(0, 1, (N, ndraw))
    
    return drw(t_obs, x, xmean, SFinf, E, N)


def draw_SFinf(lambda_RF, M_i, M_BH, size=1, randomize=True):
    """
    Get SF_\infty given AGN parameters
    """
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
    """
    Get tau given AGN parameters
    """
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
    """
    Perform inverse transform sampling on curve y(x)
    https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
    https://en.wikipedia.org/wiki/Inverse_transform_sampling
    """
    dx = np.diff(x)
    cum_values = np.zeros(x.shape)
    cum_values[1:] = np.cumsum(y*dx)/np.sum(y*dx)
    inv_cdf = interp1d(cum_values, x, fill_value='extrapolate')
    if survival:
        n_samples = int(survival)
        #print(n_samples)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def survival_sampling(y, survival, fill_value=np.nan):
    """
    Perform survival sampling on curve y given survival fraction
    """
    # Can't randomly sample, will muck up indicies
    n_samples = len(y)
    randp = np.random.rand(n_samples)
    mask_rand = (randp < survival)
    y_survive = np.full(y.shape, fill_value)
    y_survive[mask_rand] = y[mask_rand]
    return y_survive

def calcsigvar(data, error, sys_err=0.0):
    """
    Fast code variability significance of N light curves of length len
    given array of mag and magerr with shape [N, len]
    """
    
    # reshape data
    data = data.T
    error = error.T
    
    # chi^2
    wt = 1/(sys_err**2 + error**2)
    dat0 = np.sum(data*wt, axis=0)/np.sum(wt, axis=0)
    ln = len(data)
    nu = ln - 1 # dof
    chi2 = np.sum((data - dat0)**2*wt, axis=0)
    # p-value
    p = st.chi2.sf(chi2, nu)
    # sigma
    sigma_var = st.norm.ppf(1 - 0.5*p)
    return sigma_var


class DemographicModel:
    """
    Class containing methods to run model
    """
    def __init__(self, survey='lsst', workdir='work', load=False):
        # argument for input of multiple occupation fractions
        self.pars = {}
        self.workdir = workdir
        
        if load:
            with open(os.path.join(workdir, f'pars_{survey}.pkl'), 'rb') as f:
                self.pars = pickle.load(f)
        return
    
                      
    def save(self, survey='lsst'):
        """
        Save parameter data to disk
        """
        with open(os.path.join(self.workdir, f'pars_{survey}.pkl'), 'wb') as f:
            pickle.dump(self.pars, f)
        return
    
    
    def save_samples(self, samples, j):
        """
        Save array of samples to disk
        """
        for key in samples.keys():
            s = samples[key]
            if isinstance(s, u.quantity.Quantity):
                s = s.value
            np.save(os.path.join(self.workdir, f'samples_{key}_{j}'), s)
                
        
    def load_sample(self, name='z_draw', seed=None, j=0):
        """
        Load array of samples into memory
        """
        if seed is None:
            data = np.load(os.path.join(self.workdir, f'samples_{name}_{j}.npy'))
        else:
            data = np.load(os.path.join(self.workdir, f'samples_{name}_{seed}_{j}.npy'))
        return data
        

    def sample(self, nbins=10, nbootstrap=50, eta=1e4, zmax=0.1, ndraw_dim=1e7, omega=4*np.pi,
               seed_dict={'light':(lambda x: np.ones_like(x)), 'heavy':f_occ_heavyMS, 'light_stellar':(lambda x: np.ones_like(x))},
               ERDF_mode=0, log_edd_mu=-1, log_edd_sigma=0.2):
        """
        Methodology roughly follows https://ui.adsabs.harvard.edu/abs/2017ApJ...845..134W/abstract

        nbins: Number of stellar mass bins 
        nbootstrap: Number of bootstrap samples (for observational uncertainties)
        eta: Understamping factor (each MC data point corresponds to eta real galaxies)
        zmax: Maximum redshift range
        ndraw_dim: Dimensions for arrays (should be larger than number of simulated galaxies)
        occ_dict: Dictonary of occupation fractions to use
            name: occupation function
            *_stellar: If _stellar key, it's a special function that uses the stellar BHMF from Sicilia et al. 2021 + specified occupation function
        ERDF_mode: Which ERDF to adopt
            0 = Weigel 2017
            1 = active fraction
        """        
        pars = {'nbins':nbins, 'nbootstrap':nbootstrap, 'eta':eta}
        workdir = self.workdir
        
        dtype = np.float64
        
        ndraw_dim = int(ndraw_dim)
        pars['ndraw_dim'] = ndraw_dim
        pars['seed_keys'] = list(seed_dict.keys())
        
        pars['log_lambda_min'] = -8.5
        pars['log_lambda_max'] = 0.5
        
        pars['log_M_star_min'] = 4.5
        pars['log_M_star_max'] = 12.5
        
        pars['log_M_BH_min'] = 1.5
        pars['log_M_BH_max'] = 9.5
        
        print('Omega:', omega)
        print('eta:', eta)
        
        pars['ndraws'] = np.full(nbootstrap, np.nan, dtype=np.int)
        
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
        
        # M_BH - M_NSC relation from Graham 2020
        alpha_SC = np.random.normal(loc=8.22, scale=0.20, size=nbootstrap)
        beta_SC = np.random.normal(loc=2.62, scale=0.42, size=nbootstrap)
        M_SC_br = 10**7.83*u.Msun
        M_star_norm_SC = 1*u.Msun

        # Stellar Mass Function
        M_star_ = np.logspace(pars['log_M_star_min'], pars['log_M_star_max'], nbins+1, dtype=dtype)*u.Msun
        dM_star = np.diff(M_star_)
        dlogM_star = np.diff(np.log10(M_star_.value))
        M_star = M_star_[1:] + dM_star/2 # bins
        pars['M_star'] = M_star
        pars['M_star_'] = M_star_
        
        print('log M_star bins: ', np.around(np.log10(pars['M_star'].value), 2))

        # 1. Assign number of draws
        d_c_min = 0.5*u.Mpc
        pars['zmax'] = zmax
        pars['zmin'] = z_at_value(cosmo.comoving_distance, d_c_min, zmin=-1e-4, zmax=zmax+1e-4)
        V = cosmo.comoving_volume(zmax)*omega/(4*np.pi)
        pars['V'] = V
        z_samples = np.linspace(pars['zmin'], pars['zmax'])
        dz = np.diff(z_samples)
        z_bins = z_samples[1:] + dz/2

        # 4. BH Mass Function
        M_BH_ = np.logspace(pars['log_M_BH_min'], pars['log_M_BH_max'], nbins+1, dtype=dtype)*u.Msun
        dM_BH = np.diff(M_BH_)
        dlogM_BH = np.diff(np.log10(M_BH_.value))
        M_BH = M_BH_[1:] + dM_BH/2 # bins
        pars['M_BH'] = M_BH
        
        print('log M_BH bins: ', np.around(np.log10(pars['M_BH'].value), 2))

        # 5. Eddington ratio Function
        lambda_ = np.logspace(pars['log_lambda_min'], pars['log_lambda_max'], nbins+1, dtype=dtype)
        dlambda = np.diff(lambda_)
        dloglambda = np.diff(np.log10(lambda_))
        pars['lambda_Edd'] = lambda_[1:] + dlambda/2 # bins
        
        print('log lambda bins: ', np.around(np.log10(pars['lambda_Edd']), 2))

        # 6. AGN Luminosity Function
        L_ = np.logspace(34, 47, nbins+1, dtype=dtype)*u.erg/u.s
        dL = np.diff(L_)
        dlogL = np.diff(np.log10(L_.value))
        pars['L'] = L_[1:] + dL/2 # bins
        
        hists = {}
        # Number densities
        hists['n_i_M'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)
        hists['n_i_Edd'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)
        # Occupation probabililty survival sampling
        for k, seed in enumerate(seed_dict.keys()):
            hists[f'n_i_M_{seed}'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)
            hists[f'n_i_L_{seed}'] = np.full([nbootstrap, nbins], np.nan, dtype=np.int)
            
        # Bootstrap loop
        for j in tqdm(range(nbootstrap)):
                        
            # Set random seed
            np.random.seed(j)

            # 1. Assign redshifts like z \propto dV(z)
            z_draw = inv_transform_sampling(cosmo.differential_comoving_volume(z_bins).value, z_samples, n_samples=ndraw_dim)
            
            # 2. Draw from GSMF [Mpc^-3]
            phidM_blue = GSMF_blue(M_star.value, z_draw)*dM_star
            phidM_red = GSMF_red(M_star.value, z_draw)*dM_star
            if '_stellar' in seed.lower():
                phidM_wandering = BHMF_wandering(M_BH.value)
            else:
                phidM_wandering = np.zeros_like(M_BH.value)
                        
            # phi = dn / dlog M  = dN / dV / dlog M
            # Normalize
            Vred = V.to(u.Mpc**3).value / eta # Reduced comoving volume
            sf_blue = Vred * trapz((phidM_blue/dM_star).value, M_star.value)
            sf_red = Vred * trapz((phidM_red/dM_star).value, M_star.value)
            sf_wandering = Vred * trapz((phidM_wandering/dM_BH).value, M_BH.value)
            
            M_star_draw_blue = inv_transform_sampling((phidM_blue/dM_star).value, M_star_.value, survival=sf_blue)*u.Msun
            M_star_draw_red = inv_transform_sampling((phidM_red/dM_star).value, M_star_.value, survival=sf_red)*u.Msun
            M_BH_draw_wandering = inv_transform_sampling((phidM_wandering/dM_BH).value, M_BH_.value, survival=sf_wandering)*u.Msun
            
            # Number of sources
            ndraw_gal = len(M_star_draw_blue) + len(M_star_draw_red)
            ndraw_wandering = len(M_BH_draw_wandering)
            ndraw = ndraw_gal + ndraw_wandering
            
            # Get stellar mass of wandering BHs (?)
            M_star_draw_wandering = 10**((np.log10(M_BH_draw_wandering/M_SC_br) - alpha_SC[j])/beta_SC[j] + np.random.normal(0.0, 0.5, size=ndraw_wandering))*M_star_norm_SC
            # Red + blue + wandering population
            M_star_draw = np.concatenate([M_star_draw_wandering, M_star_draw_blue, M_star_draw_red])
            
            print('log N galaxies: ', np.around(np.log10(ndraw), 2))
            print(np.around(np.log10(ndraw_wandering), 2), np.around(np.log10(len(M_star_draw_blue)), 2), np.around(np.log10(len(M_star_draw_red)), 2))
            
            mask_wandering = np.concatenate([np.full(ndraw_wandering, True),
                                           np.full(len(M_star_draw_blue), False),
                                           np.full(len(M_star_draw_red), False)])
            mask_red = np.concatenate([np.full(ndraw_wandering, False),
                                       np.full(len(M_star_draw_blue), False),
                                       np.full(len(M_star_draw_red), True)])
            mask_blue = np.concatenate([np.full(ndraw_wandering, False),
                                        np.full(len(M_star_draw_blue), True),
                                        np.full(len(M_star_draw_red), False)])
                        
            if ndraw > ndraw_dim:
                print("ndraw > ndraw_dim! Try increasing ndraw_dim.")
                print('log ndraw: ', np.log10(ndraw))
                print('log ndraw_dim: ', np.log10(ndraw_dim))
                return
                
            # Shuffle
            p = np.random.permutation(ndraw)
            M_star_draw = M_star_draw[p]
            mask_wandering = mask_wandering[p]
            mask_red = mask_red[p]
            mask_blue = mask_blue[p]
            z_draw = z_draw[:ndraw][p]
            
            # Initialize arrays
            samples = {}
                            
            # Mask galaxy populations
            samples['pop'] = np.full(ndraw, np.nan, dtype=np.int)
            samples['pop'][mask_wandering] = 2
            samples['pop'][mask_red] = 1
            samples['pop'][mask_blue] = 0
                        
            pars['ndraws'][j] = ndraw
            samples['z_draw'] = z_draw
            samples['M_star_draw'] = M_star_draw
            hists['n_i_M'][j,:], _ = hist1d(M_star_draw.value[~mask_wandering], bins=M_star_.value)
            
            print('Sampling host galaxy colors')

            # Host galaxy colors 
            g_minus_r_draw = np.full(ndraw, np.nan)
            g_minus_r_draw[mask_red] = g_minus_r_model_red(M_star_draw[mask_red].value, seed=j)
            # Assume blue "host star cluster" colors for wanderers
            g_minus_r_draw[mask_blue | mask_wandering] = g_minus_r_model_blue(M_star_draw[mask_blue | mask_wandering].value, seed=j)
            samples['g-r'] = g_minus_r_draw
            
            # Host galaxy stellar mass
            M_BH_draw = np.full(ndraw, np.nan, dtype=dtype)*u.Msun
            M_BH_draw[~mask_wandering] = 10**(alpha[j] + beta[j]*np.log10(M_star_draw[~mask_wandering]/M_star_br) +
                                            np.random.normal(0.0, 0.55, size=ndraw_gal))*M_BH_norm
            M_BH_draw[mask_wandering] = M_BH_draw_wandering
            samples['M_BH_draw'] = M_BH_draw
            
            print('Sampling ERDF')
            
            # Eddington ratio distribution function (ERDF)
            if ERDF_mode == 0:
                lambda_draw = np.full(ndraw, np.nan)
                # Blue  
                xi_blue = ERDF_blue(pars['lambda_Edd']) # dN / dlog lambda
                xi_red = ERDF_red(pars['lambda_Edd']) # dN / dlog lambda
                xi_wandering = ERDF_red(pars['lambda_Edd']) # Assume these are radiatively inefficient ERDF
                norm = trapz(xi_blue + xi_red + xi_wandering, pars['lambda_Edd'])
                xi_blue = xi_blue/norm
                xi_red = xi_red/norm
                xi_wandering = xi_wandering/norm
                lambda_draw[mask_blue] = inv_transform_sampling(xi_blue/dlambda, lambda_, np.count_nonzero(mask_blue))
                lambda_draw[mask_red] = inv_transform_sampling(xi_red/dlambda, lambda_, np.count_nonzero(mask_red))
                lambda_draw[mask_wandering] = inv_transform_sampling(xi_wandering/dlambda, lambda_, np.count_nonzero(mask_wandering))
            elif ERDF_mode == 1:
                p = lambda_A(M_star_draw.value)
                lambda_draw_init = 10**np.random.normal(log_edd_mu, log_edd_sigma, ndraw)
                lambda_draw = survival_sampling(lambda_draw_init, survival=p, fill_value=1e-8)

            samples['lambda_draw'] = lambda_draw
            hists['n_i_Edd'][j,:], _ = hist1d(lambda_draw, bins=lambda_)
            
            print('Sampling occupation probability')
            
            # Occupation fraction loop
            for k, seed in enumerate(seed_dict.keys()):
                # Initialize arrays
                samples[f'M_BH_draw_{seed}'] = np.full(ndraw, np.nan, dtype=dtype)*u.Msun
                samples[f'L_draw_{seed}'] = np.full(ndraw, np.nan, dtype=dtype)*u.erg/u.s
                
                # Mask artificial stellar BHMF "host galaxies" unless "*_stellar" in key
                if '_stellar' in seed.lower():
                    mask_seed = mask_wandering | mask_blue | mask_red
                else:
                    mask_seed = ~mask_wandering
                
                # Occupation probabililty survival sampling
                p = seed_dict[f'{seed}'](M_star_draw.value[mask_seed])
                M_BH_draw_seed = survival_sampling(M_BH_draw.value[mask_seed], survival=p, fill_value=0.0)*u.Msun
                samples[f'M_BH_draw_{seed}'][mask_seed] = M_BH_draw_seed
                hists[f'n_i_M_{seed}'][j,:], _ = hist1d(M_BH_draw_seed.value, bins=M_BH_.value)
                
                print('log N BHs: ', np.around(np.log10(len(M_BH_draw_seed)), 2))

                # 6. AGN Luminosity Function
                L_draw_seed = lambda_draw[mask_seed] * 1.26e38 * M_BH_draw_seed.to(u.Msun).value * u.erg/u.s
                samples[f'L_draw_{seed}'][mask_seed] = L_draw_seed
                hists[f'n_i_L_{seed}'][j,:], _ = hist1d(L_draw_seed.value, bins=L_.value)
                        
            # Save the results to free up memory
            self.save_samples(samples, j)
                
        # Save histograms
        for key in hists.keys():
            # Correct for numerical factor and save the results on final iteration only
            np.save(os.path.join(workdir, f'hist_{key}'), hists[key]*int(eta))
        
        self.pars = pars
        self.samples_keys = samples.keys()
        return
        
        
    def sample_sed_grid(self, w0=1e-3, w1=1e8, band='SDSS_g', model_sed_name='optxagnf', nbins=9, save_fits=True, load_fits=False,
                       sed_pars={'bh_mass':1e8,'dist_c':30.0,'lambda_edd':np.log10(0.1),'spin':0,'r_cor':100,
                                 'log_r_out':-1,'kT_e':0.23,'tau':11,'gamma':2.2,'f_pl':0.05,'z':0.007,'norm':1}):
        
        pars = self.pars
        samples_keys = self.samples_keys
        workdir = self.workdir
                
        ndraws = pars['ndraws']
        ndraw_dim = int(np.max(ndraws))
        
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
        
        energies = model_sed.energies(0)[::-1]*u.keV # Why N-1?
        wav_RF = (energies[:-1]).to(u.nm, equivalencies=u.spectral())
        
        model_sed_riaf = {'wav':wav_RF.to(u.nm).value, 'dir':'/home/colinjb2/riaf-sed'}
        
        # Initalize the grid
        # hard-coding some things to avoid boundary conditions in the SEDs
        x = np.logspace(2, 9, nbins)
        y = np.logspace(-8, 0, nbins)
        y_agn = y[y>=1e-3]
        y_riaf = y[y<1e-3]
        z = np.linspace(pars['zmin'], pars['zmax'], nbins)
        
        print(f'Creating SED grid in band {band}')
        
        if load_fits:
            with fits.open('sed_grid.fits') as hdul:
            
                data0 = hdul[0].data
                nuf_nu = (10**data0) * u.erg/u.s/u.cm**2

                M_i_model = hdul[1].data

                shape = np.shape(nuf_nu)

                data1 = hdul[2].data
                x = 10**data1['log_M_BH']
                y = 10**data1['log_LAMBDA_EDD']
                z = data1['Z']

                d_L = cosmo.luminosity_distance(z).to(u.cm)

                wav = (10**hdul[3].data['log_WAV']) * u.nm
                bandpass = lib[band]

                # Get L_band
                f_lambda = nuf_nu/wav

                f_lambda_band = np.full(shape[:-1], np.nan) * u.erg/u.s/u.cm**2/u.AA
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):
                            f_lambda_band[i,j,k] = bandpass.get_flux(wav, f_lambda[i,j,k,:])

                f_band = f_lambda_band * bandpass.lpivot
                L_band_model = (f_band * 4*np.pi*d_L**2).value
                        
        else:
            
            print('Generating AGN SEDs')
            # Get AGN luminosity in band and M_i(z=2)
            vget_AGN_flux = np.vectorize(get_AGN_flux, otypes=[np.float,np.float,np.float,np.float,np.ndarray])

            X, Y, Z = np.meshgrid(x, y_agn, z, indexing='ij', sparse=True)
            _, _, L_band_model_AGN, _, nuf_nu_AGN = vget_AGN_flux(model_sed, M_BH=X, lambda_Edd=Y, z=Z, band=band)

            X, Y = np.meshgrid(x, y_agn, indexing='ij', sparse=True)
            M_i_model_AGN, _, _, _, _ = vget_AGN_flux(model_sed, M_BH=X, lambda_Edd=Y, z=2.0, band='SDSS_i')
            
            print('Generating RIAF SEDs')
            # Get RIAF luminosity in band and M_i(z=2)
            vget_RIAF_flux = np.vectorize(get_RIAF_flux, otypes=[np.float,np.float,np.float,np.float,np.ndarray])

            X, Y, Z = np.meshgrid(x, y_riaf, z, indexing='ij', sparse=True)
            _, _, L_band_model_RIAF, _, nuf_nu_RIAF = vget_RIAF_flux(model_sed_riaf, M_BH=X, lambda_Edd=Y, z=Z, band=band)

            X, Y = np.meshgrid(x, y_riaf, indexing='ij', sparse=True)
            M_i_model_RIAF, _, _, _, _ = vget_RIAF_flux(model_sed_riaf, M_BH=X, lambda_Edd=Y, z=2.0, band='SDSS_i')
                        
            L_band_model = np.concatenate([L_band_model_RIAF, L_band_model_AGN], axis=1)
            M_i_model = np.concatenate([M_i_model_RIAF, M_i_model_AGN], axis=1)
            nuf_nu = np.concatenate([nuf_nu_RIAF, nuf_nu_AGN], axis=1)
            y = np.concatenate([y_riaf, y_agn])

                
        # Create interpolator objects
        fn_L_band_model = RegularGridInterpolator((np.log10(x), np.log10(y), z), np.log10(L_band_model),
                                                  bounds_error=False, fill_value=None)
        fn_M_i_model = RegularGridInterpolator((np.log10(x), np.log10(y)), M_i_model,
                                               bounds_error=False, fill_value=None)
                
        for k, seed in enumerate(pars['seed_keys']):
            
            print(f'Sampling SEDs with seeding mechanism {seed}')
        
            # Sample the grid at each source
            for j in tqdm(range(nbootstrap)):
                
                samples = {}
                ndraw = int(ndraws[j])
                
                # MUST initialize L_AGN to 0, so host-galaxy mag is finite even when unoccupied!
                samples[f'L_{band}_{seed}'] = np.full(ndraw, 0.0)*u.erg/u.s
                samples[f'M_i_{seed}'] = np.full(ndraw, np.nan)
                
                L_draw_seed = np.load(os.path.join(self.workdir, f'samples_L_draw_{seed}_{j}.npy'))*u.erg/u.s
                M_BH_draw_seed = np.load(os.path.join(self.workdir, f'samples_M_BH_draw_{seed}_{j}.npy'))*u.Msun
                lambda_draw = np.load(os.path.join(self.workdir, f'samples_lambda_draw_{j}.npy'))
                z_draw = np.load(os.path.join(self.workdir, f'samples_z_draw_{j}.npy'))
                
                mask_occ = M_BH_draw_seed.value > 0.0
                
                xj = np.log10(M_BH_draw_seed.value[mask_occ])
                yj = np.log10(lambda_draw[mask_occ])
                zj = z_draw[:ndraw][mask_occ]
                points_3 = np.array([xj, yj, zj]).T
                points_2 = np.array([xj, yj]).T
                # Need to set the luminosity to 0 if M_BH=0
                samples[f'L_{band}_{seed}'][mask_occ] = (10**fn_L_band_model(points_3))*u.erg/u.s
                samples[f'M_i_{seed}'][mask_occ] = fn_M_i_model(points_2)
                
                # Obscured fraction
                p = 1 - lambda_obs(L_draw_seed[mask_occ].value, seed=j)
                L_band = samples[f'L_{band}_{seed}'][mask_occ]
                L_band_obs = survival_sampling(L_band.value, survival=p, fill_value=0.0)*u.erg/u.s
                samples[f'L_{band}_{seed}'][mask_occ] = L_band_obs
                samples[f'M_i_{seed}'][mask_occ][L_band_obs.value==0] = np.nan
                
                # Save the results to free up memory
                self.save_samples(samples, j)
                
        
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
            log_nuf_nu = np.log10(np.array(nuf_nu.tolist()))
            hdu0 = fits.PrimaryHDU(log_nuf_nu)
              
            M_i_model = np.array(M_i_model.tolist())
            hdu1 = fits.ImageHDU(M_i_model)

            table_hdu = fits.HDUList([hdu0, hdu1, table_hdu0, table_hdu1])
            table_hdu.writeto('sed_grid.fits', overwrite=True)
            # M_i can be obtained using the approximation M_i = 125 - 3.3 log(L_bol / erg s^−1)

        return
    
    
    def sample_SF_tau(self, j, seed, t_obs, pm_prec=pm_prec, band='SDSS_g', SFinf_small=1e-8, m_5=25.0):

        ndraws = self.pars['ndraws']
        ndraw = int(ndraws[j])
        workdir = self.workdir

        # Load arrays
        z = np.load(os.path.join(self.workdir, f'samples_z_draw_{j}.npy'))
        M_BH = np.load(os.path.join(self.workdir, f'samples_M_BH_draw_{seed}_{j}.npy'))*u.Msun
        M_star = np.load(os.path.join(self.workdir, f'samples_M_star_draw_{j}.npy'))*u.Msun
        g_minus_r = np.load(os.path.join(self.workdir, f'samples_g-r_{j}.npy'))
        L_band_AGN = np.load(os.path.join(self.workdir, f'samples_L_{band}_{seed}_{j}.npy'))*u.erg/u.s
        L_bol_AGN = np.load(os.path.join(self.workdir, f'samples_L_draw_{seed}_{j}.npy'))*u.erg/u.s
        M_i_AGN = np.load(os.path.join(self.workdir, f'samples_M_i_{seed}_{j}.npy'))

        # Initialize arrays
        samples = {}
        samples[f'm_{band}_{seed}'] = np.full(ndraw, np.nan)
        samples[f'SFinf_{band}_{seed}'] = np.full(ndraw, np.nan)
        samples[f'tau_RF_{band}_{seed}'] = np.full(ndraw, np.nan)

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

        """
        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(M_star.value, (M_star/(1*u.Msun) / 10**(b*g_minus_r + a + color_var)), s=.1)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.show()

        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(M_star.value, g_minus_r, s=1)
        ax.set_xscale('log')
        plt.show()

        fig = plt.figure()
        ax = plt.gca()
        ax.scatter(M_star.value, 10**(b*g_minus_r + a + color_var), s=1)
        ax.scatter(M_star.value, 10**(b*g_minus_r + a), s=1)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.show()
        """
        
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
        elif band == 'SDSS_g':
            f_band_host = L_band_host / (4*np.pi*d_L**2)
            f_lambda_band_host = (f_band_host / lib[band].lpivot).to(u.erg/u.s/u.cm**2/u.AA)
            # K-correction
            #print(k_corr(z, g_minus_r))
            m_band_host = -2.5*np.log10(f_lambda_band_host.value) - lib[band].AB_zero_mag #+ k_corr(z, g_minus_r)
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
        samples[f'm_{band}_{seed}'] = m_band

        # Draw SF_\infty and tau (rest-frame)
        # In M10, M_i is basically a proxy for L_bol, so we need to use the Shen relation
        # even for IMBHs, to preserve the linearity in the extrapolation of this relation
        M_i_AGN = 90 - 2.5*np.log10(L_bol_AGN.to(u.erg/u.s).value)

        SFinf = draw_SFinf(lambda_RF.to(u.AA).value, M_i_AGN, M_BH.value, size=ndraw)
        tau = draw_tau(lambda_RF.to(u.AA).value, M_i_AGN, M_BH.value, size=ndraw)

        # Host-galaxy dilution
        dL = L_band_AGN*np.log(10)/2.5*SFinf #* 1.3 # The 1.3 is a fudge factor to normalize with the qsos
        SFinf = 2.5/np.log(10)*dL/(L_band_AGN + L_band_host)
        
        samples[f'SFinf_{band}_{seed}'] = SFinf
        samples[f'tau_RF_{band}_{seed}'] = tau

        # Light curves
        mask_small = (SFinf < SFinf_small) | (M_BH < 1e2*u.Msun) | (m_band > m_5+1)
        
        # Save the results to free up memory
        self.save_samples(samples, j)

        return SFinf[~mask_small], tau[~mask_small], z[~mask_small], m_band[~mask_small], mask_small
    
    
    def sample_light_curve(self, j, seed, SFinf, tau, z, m_band, mask_small, t_obs,
                           pm_prec=pm_prec, dt_min=10, band='SDSS_g', SFinf_small=1e-8, m_5=25.0):
        
        ndraws = self.pars['ndraws']
        ndraw = int(ndraws[j])
        workdir = self.workdir
        
        t_obs_dense = np.arange(np.min(t_obs), np.max(t_obs), dt_min)
        t_rest_dense = np.arange(np.min(t_obs), np.max(t_obs), dt_min)
        
        shape = np.count_nonzero(~mask_small)
        #t_obs_dense_shaped = (np.array([t_obs_dense]*shape)).T
        t_rest_dense_shaped = (np.array([t_rest_dense]*shape)).T / (1 + z)

        # Simulate light curves (clip tau to avoid numerical issues if tau << dt_min)
        lcs = simulate_drw(t_rest_dense_shaped, np.clip(tau, 2*dt_min, None), z, m_band, SFinf).T
        f = interp1d(t_obs_dense, lcs, fill_value='extrapolate')
        mag = f(t_obs)
        samples = {}
        samples[f'std_{seed}'] = np.full(ndraw, np.nan)
        samples[f'sigvar_{seed}'] = np.full(ndraw, np.nan)
        samples[f'lc_{band}_{seed}_idx'] = np.full(ndraw, np.arange(ndraw), dtype=np.int)
        
        #samples[f'lc_{band}_{seed}'] = mag
        samples[f'lc_{band}_{seed}_idx'][mask_small] = -1

        # Calculate variability significance
        indx = samples[f'lc_{band}_{seed}_idx'][~mask_small]
        indx = indx[np.isfinite(indx)].astype(np.int)

        # Add uncertainty
        magerr = pm_prec(mag)
        mag_obs = np.clip(np.random.normal(mag, magerr), 0, 30)
        
        samples[f'std_{seed}'][indx] = np.std(mag_obs, axis=1)
        samples[f'sigvar_{seed}'][indx] = calcsigvar(mag_obs, magerr)

        """
        for l, v in enumerate(indx):
            try:
                r = qso_fit(t_obs, mag_obs[l], magerr[l])
                samples[f'sigvar_{seed}'][v] = r['signif_vary']
                samples[f'sigqso_{seed}'][v] = r['signif_qso']
            except:
                # why the nans?
                #print(mag_obs[l])
                pass
        """

        # Save the results to free up memory
        self.save_samples(samples, j)
        return
    
    
    def sample_light_curves(self, t_obs, pm_prec=pm_prec, dt_min=10, band='SDSS_g', SFinf_small=1e-8, m_5=25.0):
        
        pars = self.pars
        workdir = self.workdir
        ndraws = pars['ndraws']
        
        ndraw_dim = int(np.max(pars['ndraws']))
        nbootstrap = pars['nbootstrap']
        
        pars['lc_t_obs'] = t_obs
                        
        for k, seed in enumerate(pars['seed_keys']):
            
            print(f'Sampling light curves with seeding mechanism {seed}')
            
            for j in tqdm(range(nbootstrap)):
                
                SFinf, tau, z, m_band, mask = self.sample_SF_tau(j, seed, t_obs, pm_prec=pm_prec, band=band,
                                                                 SFinf_small=SFinf_small, m_5=m_5)
                
                self.sample_light_curve(j, seed, SFinf, tau, z, m_band, mask, t_obs, pm_prec=pm_prec, dt_min=dt_min,
                                        band=band, SFinf_small=SFinf_small, m_5=m_5)
                                
        return
            
        
    def plot(self, seed_dict={'light':(lambda x: np.ones_like(x)), 'heavy':f_occ_heavyMS, 'light_stellar':(lambda x: np.ones_like(x))}, seed_colors=None, seed_markers=None, figsize=(3*6, 2*5), moct=np.nanmean, n_bin_min=10):
    
        import matplotlib.ticker as ticker

        fig, axs = plt.subplots(2, 3, figsize=figsize)
        axs = axs.flatten()
        
        pars = self.pars
        ndraws = pars['ndraws']
        workdir = self.workdir
        
        ndraw_dim = int(np.max(ndraws))
        
        V = pars['V']
        
        n_i_M = np.load(os.path.join(workdir, f'hist_n_i_M.npy'))
        n_i_Edd = np.load(os.path.join(workdir, f'hist_n_i_Edd.npy'))
        #n_i_M = hists['n_i_M'] #[:,:ndraw_dim]
        #n_i_Edd = hists['n_i_Edd'] #[:,:ndraw_dim]
        
        # Clean
        #n_i_M[:, n_i_M[0] < n_bin_min] = np.nan
        #n_i_Edd[:, n_i_Edd[0] < n_bin_min] = np.nan
                        
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
        axs[0].fill_between(M_star.value, (np.nanpercentile(n_i_M, 16, axis=0)/dlogM_star/V).value,
                            (np.nanpercentile(n_i_M, 84, axis=0)/dlogM_star/V).value, color='k', alpha=0.5)
        axs[0].scatter(M_star.value, moct(n_i_M, axis=0)/dlogM_star/V, lw=3, color='k')

        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel(r'$M_{\star}\ (M_{\odot})$', fontsize=18)
        axs[0].set_ylabel(r'$\phi(M_{\star})$ (dex$^{-1}$ Mpc$^{-3}$)', fontsize=18)
        axs[0].set_xlim([1e6, 4e11])
        axs[0].set_ylim([1e-5, 7e-1])

        # Real data
        # https://ui.adsabs.harvard.edu/abs/2012MNRAS.421..621B
        x = np.array([6.25,6.75,7.10,7.30,7.50,7.70,7.90,8.10,8.30,8.50,8.70,8.90,9.10,9.30,9.50,9.70,9.90,\
                      10.1,10.3,10.5,10.7,10.9,11.1,11.3,11.5,11.7,11.9])
        y = np.array([31.1,18.1,17.9,43.1,31.6,34.8,27.3,28.3,23.5,19.2,18.0,14.3,10.2,9.59,7.42,6.21,5.71,5.51,5.48,5.12,\
                      3.55,2.41,1.27,0.33,0.042,0.021,0.042])
        axs[0].scatter(10**x, y*1e-3, c='r', marker='x', label='GAMA')

        # 3. BH occupation fraction
        nbootstrap = pars['nbootstrap']

        for k, seed in enumerate(pars['seed_keys']):
            
            if '_stellar' in seed.lower():
                continue
            
            f = np.zeros([nbootstrap, len(M_star)])
            for j in range(nbootstrap):
                f[j,:] = seed_dict[seed](M_star.value)
                
            axs[1].fill_between(M_star.value, np.percentile(f, 16, axis=0), np.percentile(f, 84, axis=0),
                                color=seed_colors[seed], alpha=0.5)
            axs[1].scatter(M_star, moct(f, axis=0), lw=3, color=seed_colors[seed], marker=seed_markers[seed])
        
        axs[1].set_xlabel(r'$M_{\star}\ (M_{\odot})$', fontsize=18)
        axs[1].set_ylabel(r'$\lambda_{\rm{occ}}$', fontsize=18)
        axs[1].set_xscale('log')
        axs[1].set_xlim([1e6, 4e11])
        axs[1].set_ylim([0, 1.1])
                                  
        # Bootstrap loop
        for j in range(1):
            pop = self.load_sample(f'pop', j=j)
            mask_pop = pop < 2
            
            M_star_draw = self.load_sample('M_star_draw', j=j)[mask_pop]
            M_BH_draw = self.load_sample(f'M_BH_draw_{seed}', j=j)[mask_pop]
        
            # M_BH - M_star
            bin_med, _, _ = st.binned_statistic(M_star_draw.flatten(), M_BH_draw.flatten(), np.nanmedian, bins=M_star_)
            bin_hi, _, _ = st.binned_statistic(M_star_draw.flatten(), M_BH_draw.flatten(), lambda x: np.nanpercentile(x, 84), bins=M_star_)
            bin_lo, _, _ = st.binned_statistic(M_star_draw.flatten(), M_BH_draw.flatten(), lambda x: np.nanpercentile(x, 16), bins=M_star_)
        
        axs[2].scatter(M_star, bin_med, lw=3, color='k')
        axs[2].fill_between(M_star.value, bin_hi, bin_lo, color='k', alpha=0.5)
        axs[2].set_xscale('log')
        axs[2].set_yscale('log')
        axs[2].set_xlim([1e6, 1e11])
        axs[2].set_ylim([1e2, 1e8])
        axs[2].set_ylabel(r'$M_{\rm{BH}}\ (M_{\odot})$', fontsize=18)
        axs[2].set_xlabel(r'$M_{\rm{\star}}\ (M_{\odot})$', fontsize=18)
        
        # BH mass function
        for k, seed in enumerate(pars['seed_keys']):
            n_i_M_seed = np.load(os.path.join(workdir, f'hist_n_i_M_{seed}.npy'))
            axs[3].fill_between(M_BH.value,
                                (np.nanpercentile(n_i_M_seed, 16, axis=0)/dlogM_BH/V).value,
                                (np.nanpercentile(n_i_M_seed, 84, axis=0)/dlogM_BH/V).value, 
                                color=seed_colors[seed], alpha=0.5)
            axs[3].scatter(M_BH, moct(n_i_M_seed, axis=0)/dlogM_BH/V,
                           lw=3, color=seed_colors[seed], marker=seed_markers[seed])

        axs[3].set_xlabel(r'$M_{\rm{BH}}\ (M_{\odot})$', fontsize=18)
        axs[3].set_ylabel(r'$\phi(M_{\rm{BH}})$ (dex$^{-1}$ Mpc$^{-3}$)', fontsize=18)
        axs[3].set_xscale('log')
        axs[3].set_yscale('log')
        axs[3].set_xlim([1e2, 1e8])
        axs[3].set_ylim([1e-4, 1e4])

        # 5. Eddington ratio distribution (ERDF) function
        norm = np.nansum(moct(n_i_Edd, axis=0)*dloglambda)
        axs[4].scatter(lambda_Edd, moct(n_i_Edd, axis=0)/dloglambda/norm, lw=3, color='k')
        axs[4].fill_between(lambda_Edd, (np.nanpercentile(n_i_Edd, 16, axis=0)/dloglambda/norm),
                            (np.nanpercentile(n_i_Edd, 84, axis=0)/dloglambda/norm), color='k', alpha=0.5)

        axs[4].set_xlabel(r'$\lambda_{\rm{Edd}}$', fontsize=18)
        axs[4].set_ylabel(r'$\xi$ (dex$^{-1}$)', fontsize=18)
        axs[4].set_xscale('log')
        axs[4].set_yscale('log')
        axs[4].set_xlim([5e-8, 1e0])
        axs[4].set_ylim([1e-4, 1e1])
        
        # 6. AGN Luminosity Function
        for k, seed in enumerate(pars['seed_keys']):
            
            n_i_L_seed = np.load(os.path.join(workdir, f'hist_n_i_L_{seed}.npy'))
            #n_i_L_seed[:,n_i_L_seed[0] < n_bin_min] = np.nan
            
            mask_L = L.value < 1e45 ##
            
            axs[5].scatter(L[mask_L], moct(n_i_L_seed, axis=0)[mask_L]/dlogL/V,
                           lw=3, color=seed_colors[seed], marker=seed_markers[seed])
            axs[5].fill_between(L[mask_L].value, (np.nanpercentile(n_i_L_seed, 16, axis=0)[mask_L]/dlogL/V).value,
                                (np.nanpercentile(n_i_L_seed, 84, axis=0)[mask_L]/dlogL/V).value,
                                color=seed_colors[seed], alpha=0.5)
        
        def phi_shen(L):
            # z = 0.2
            L_star = ((10**11.275)*u.Lsun).to(u.erg/u.s).value
            phi_star = 10**-4.240
            y1 = 0.787
            y2 = 1.713
            return phi_star / ( (L/L_star)**y1 + (L/L_star)**y2 )
            
        x = np.logspace(42.5, 47)
        #axs[5].scatter(x, phi_shen(x), c='r', marker='x')

        # Store variables for later
        axs[5].set_xlabel(r'$L_{\rm{bol}}$ (erg s$^{-1}$)', fontsize=18)
        axs[5].set_ylabel(r'$\phi(L_{\rm{bol}})$ (dex$^{-1}$ Mpc$^{-3}$)', fontsize=18)
        axs[5].set_xscale('log')
        axs[5].set_yscale('log')
        axs[5].set_xlim([1e36, 1e45])
        axs[5].set_ylim([1e-8, None])

        import string, matplotlib
        labels = list(string.ascii_lowercase)

        for i, ax in enumerate(axs):

            ax.text(0.02, 0.91, f'({labels[i]})', transform=ax.transAxes, fontsize=16, weight='bold', zorder=10)
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