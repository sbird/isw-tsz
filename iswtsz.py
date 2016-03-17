"""Module to implement the ISW and tSZ amplitudes from Cyril's mathematica notebook"""

import math
import numpy as np
import scipy
import scipy.integrate
import scipy.special
import halo_mass_function as hm
import concentration

def hzoverh0(a, omegam0):
    """ returns: H(a) / H0  = [omegam/a**3 + (1-omegam)]**0.5 """
    return np.sqrt(omegam0*np.power(a,-3) + (1.-omegam0))

def _lingrowthintegrand(a,omegam0):
    """ (e.g. eq. 8 in lukic et al. 2008)   returns: da / [a*H(a)/H0]**3 """
    return np.power((a * hzoverh0(a,omegam0)),-3)

def _ang_diam_integrand(aa, omegam0):
    """Integrand for the ang. diam distance: 1/a^2 H(a)"""
    return 1./(aa**2*hzoverh0(aa, omegam0))

def _conf_time_integrand(aa, omegam0):
    """Integrand for the ang. diam distance: 1/a^2 H(a)"""
    return 1./hzoverh0(aa, omegam0)

def conformal_time(aa, omegam0):
    """Conformal time to scale factor a: int(1/H da).
    This is dimensionless; to get dimensionful value divide by H0."""
    conf, err = scipy.integrate.quad(_conf_time_integrand, aa, 1., omegam0)
    if err/(conf+0.01) > 0.1:
        raise RuntimeError("Err in angular diameter: ",err)
    return conf

def angular_diameter(zz, omegam0, hub=0.7):
    """The angular diameter distance to some redshift in units of comoving Mpc."""
    aa = 1./(1+zz)
    #speed of light in km/s
    light = 2.99e5
    #Dimensionless comoving distance
    comoving,err = scipy.integrate.quad(_ang_diam_integrand, aa, 1, omegam0)
    if err/(comoving+0.01) > 0.1:
        raise RuntimeError("Err in angular diameter: ",err)
    #angular diameter distance
    H0 = hub * 100 # km/s/Mpc
    return comoving * light * aa / H0

class TSZHalo(object):
    """Extends the NFW halo calculation with equations for a tSZ profile."""
    def __init__(self, omegam0 = 0.27, omegab0=0.0449, hubble = 0.7, sigma8=0.8):
        self.omegam0 = omegam0
        self.omegab0 = omegab0
        self.hubble = hubble
        #km/s/Mpc
        self.H0 = 100* self.hubble
        #speed of light in km/s
        self.light = 2.99e5
        self.overden = hm.Overdensities(0,omegam0, omegab0,1-omegam0,hubble, 0.97,sigma8,log_mass_lim=(12,15))
        self._reds = np.linspace(0,9,25)
        self._growths = np.array([self._generate_lingrowthfac(zz) for zz in self._reds])
        #Normalise to z=0.
        self._growths/=self._growths[0]
        self.lingrowthfac=scipy.interpolate.InterpolatedUnivariateSpline(self._reds,self._growths)
        self.conc_model = concentration.LudlowConcentration(self.lingrowthfac)
        self._conformal = np.array([conformal_time(1/(1+zz), omegam0) for zz in self._reds])
        self.conformal_time = scipy.interpolate.InterpolatedUnivariateSpline(self._reds,self._conformal)

    def mass_function(self, sigma):
        """Tinker et al. 2008, eqn 3, Delta=200 # Delta=200"""
        A = 0.186
        a = 1.47
        b = 2.57
        c = 1.19
        return A * ( np.power((sigma / b), -a) + 1) * np.exp(-1 * c / sigma / sigma)

    def dndm_z(self, mass, zz):
        """Returns the halo mass function dn/dM in units of h^4 M_sun^-1 Mpc^-3
        Requires mass in units of M_sun /h """
        # Mean matter density at redshift z in units of h^2 Msolar/Mpc^3
        # This is the comoving density at redshift z in units of h^-1 M_sun (Mpc/h)^-3 (comoving)
        # Compute as a function of the critical density at z=0.
        rhom = self.overden.rhocrit(0) * self.omegam0
        growth = self.lingrowthfac(zz)
        sigma = self.overden.sigmaof_M(mass)*growth
        mass_func = self.mass_function(sigma)
        dlogsigma = self.overden.log_sigmaof_M(mass)*growth/mass
        #We have dn / dM = - d ln sigma /dM rho_0/M f(sigma)
        dndM = - dlogsigma*mass_func/mass*rhom
        return dndM

    def _generate_lingrowthfac(self, red):
        """
        returns: linear growth factor, b(a) normalized to 1 at z=0, good for flat lambda only
        a = 1/1+z
        b(a) = Delta(a) / Delta(a=1)   [ so that b(z=0) = 1 ]
        (and b(a) [Einstein de Sitter, omegam=1] = a)

        Delta(a) = 5 omegam / 2 H(a) / H(0) * integral[0:a] [da / [a H(a) H0]**3]
        equation  from  peebles 1980 (or e.g. eq. 8 in lukic et al. 2008) """

        a = 1/(1.+red)

        omegam0 = self.omegam0
        ## 1st calc. for z=z
        lingrowth,err = scipy.integrate.quad(_lingrowthintegrand,0.,a, (omegam0))
        if err/lingrowth > 0.1:
            raise RuntimeError("Err in linear growth: ",err)
        lingrowth *= 5./2. * omegam0 * hzoverh0(a,omegam0)

        return lingrowth

    def concentration(self,mass, zz):
        """Compute the concentration for a halo mass"""
        nu = 1.686/self.overden.sigmaof_M(mass)/self.lingrowthfac(zz)
        return self.conc_model.concentration(nu, zz)

    def R200(self, mass):
        """Get the virial radius in Mpc/h for a given mass in Msun/h"""
        rhoc = self.rhocrit(0)
        #in kg
        solarmass = 1.98855e30
        #1 Mpc in m
        Mpc = 3.086e22
        #Virial radius R200 in Mpc from the virial mass
        R200 = np.cbrt(3 * mass * solarmass / (4* math.pi* 200 * rhoc))/Mpc
        return R200

    def rhocrit(self, zz):
        """Critical density at redshift of the snapshot. Units are kg m^-3."""
        #Newtons constant in units of m^3 kg^-1 s^-2
        gravity = 6.67408e-11
        #scale factor
        aa = 1./(1+zz)
        #Hubble factor (~70km/s/Mpc) at z=0 in s^-1
        hubble = self.hubble*3.24077929e-18
        hubz2 = (self.omegam0/aa**3 + 1-self.omegam0) * hubble**2
        #Critical density at redshift in units of kg m^-3
        rhocrit = 3 * hubz2 / (8*math.pi* gravity)
        return rhocrit

    def Rs(self, mass, zz):
        """Scale radius of the halo in Mpc/h"""
        conc = self.concentration(mass,zz)
        return self.R200(mass)/conc

    def tsz_per_halo(self, M, zz,kk):
        """The 2D fourier transform of the projected Compton y-parameter, per halo.
        Eq 2 of Komatsu and Seljak 2002, slightly modified to remove some of the dimensionless parameters.
        Units are Mpc/h^2."""
        #Upper limit is the virial radius.
        integrated,err = scipy.integrate.quad(self._ygas3d_integrand, 0, self.R200(M), (kk, M,zz))
        if err/integrated > 0.1:
            raise RuntimeError("Err in linear growth: ",err)
        #Units:  Mpc*2
        return 4 * math.pi * integrated

    def ll_s(self, M, zz):
        """Comoving angular wavenumber."""
        return angular_diameter(zz, self.omegam0, self.hubble) / self.Rs(M,zz)

    def y3d(self, x, mass,zz):
        """Electron pressure profile from K&S 2002 eq 7. Units of 1 / Mpc."""
        return 1.04e-4 * (55/50.) * self.Pgas(x, mass,zz)

    def gamma(self, conc):
        """Polytropic index for a cluster from K&S 02, eq. 17"""
        return 1.137 + 8.94e-2 * np.log(conc/5) - 3.68e-3 * (conc-5)

    def eta0(self, conc):
        """Mass-temperature normalisation factor at the center, eta(0)."""
        return 2.235 + 0.202 * (conc - 5) - 1.16e-3*(conc-5)**2

    def ygas(self, x, conc):
        """Dimensionless part of the K&S pressure profile, eq. 15"""
        #K&S 02 eq. 17 and 18, fitting formulae.
        gamma = self.gamma(conc)
        eta0 = self.eta0(conc)
        BB = 3./eta0 * (gamma - 1)/gamma / (np.log(1+conc)/conc - 1/(1+conc))
        return np.power(1 - BB * (1- np.log(1+x)/x), 1./(gamma-1))

    def Pgas(self, x, mass, zz):
        """Dimensionless part of gas pressure profile, K&S 02 eq 8.
        Other choices are possible (indeed preferred)!"""
        #Boltzmann constant in eV/K
        kboltz = 8.6173324e-5
        conc = self.concentration(mass, zz)
        gamma = self.gamma(conc)
        #In M_sun / Mpc^3, as the constant below.
        #Eq. 21
        rhogas0 = 7.96e13 * (self.omegab0 * self.hubble**2 / self.omegam0)
        rhogas0 *=  (mass / 1e15)/self.R200(mass)**3
        rhogas0 *= conc**3 / (np.log(1+conc) - conc/(1+conc)) /(conc**2*(1+conc)**2*self.ygas(conc, conc))
        #Tgas at 0.: eq. 19.
        Tgas0 = 8.8 * self.eta0(conc) * (mass / 1e15)/self.R200(mass)
        Pgas0 = rhogas0 / 1e14 * Tgas0 * kboltz/1e3/8
        return Pgas0 * self.ygas(x, conc)**gamma

    def _ygas3d_integrand(self, rr, kk, mass,zz):
        """Integrand of the TSZ profile from Komatsu & Seljak 2002 eq. 2"""
        return scipy.special.j0(rr*kk) * self.y3d(rr/self.Rs(mass, zz), mass,zz) * rr * rr

    def isw_hh_l(self, kk, l):
        """The change in the CMB temperature from the late-time (dark energy) ISW effect.
        Eq. 7 of Taburet 2010, 1012.5036"""
        #Upper limit for the integral should be the conformal distance to recombination.
        #In practice the integrand is zero for matter domination, so it doesn't really matter.
        #This has units of 1 Mpc^3
        integrated = scipy.integrate.quad(self._isw_integrand,0.1,1, (kk,l))
        #units:     mpc^-2                          mpc^2
        return 3 * self.H0**2 / self.light**2 * self.omegam0 / kk**2 * integrated

    def _isw_integrand(self, aa, kk, l):
        """Integrand for the ISW calculation above.
        (da)  d/da(D+/a) j_l(k*r). Dimensionless"""
        #Comoving distance to this scale factor.
        zz = 1./aa - 1
        rr = self.light / self.H0 * self.conformal_time(zz)
        #d/da (D+/a) = 5/2 omega_M / (H^2 a^4) + D+ /a ( H'/H - 1 /a)
        #Zero if H  = omega_M a^-3/2, D+ ~ a as H'/H ~ -3/2 / a
        H2 = hzoverh0(aa, self.omegam0)**2
        Hprime = 1.5 * self.omegam0 / aa**4 / H2
        Dplusda = 2.5 * self.omegam0 / (H2* aa**4) - self.lingrowthfac(aa) / aa * ( Hprime + 1/aa)
        return Dplusda * scipy.special.sph_jn(l, kk * rr)

    def bias(self,mass, zz):
        """Formula for the bias of a halo, from Sheth-Tormen 1999."""
        delta_c = 1.686
        nu = 1.686/self.overden.sigmaof_M(mass)/self.lingrowthfac(zz)
        bhalo = (1 + (0.707*nu**2 -1)/ delta_c) + 2 * 0.3 * delta_c / (1 + (0.707 * nu**2)**0.3)
        return bhalo

    def tsz_mass_integral(self, zz, kk, mass=(1e12, 1e15)):
        """Formula for the mass integral of the tSZ power, Taburet 2010, 1012.5036.
        Units of h/Mpc"""
        (integrated, err) = scipy.integrate.quad(self._tsz_mass_integrand, mass[0], mass[1], (zz, kk))
        if err/(integrated+0.01) > 0.1:
            raise RuntimeError("Err in tsz mass integral: ",err)
        return integrated

    def _tsz_mass_integrand(self, MM, zz, kk):
        """Integrand for the tSZ mass integral. Units of h/Mpc h/Msun"""
        y3d = self.tsz_per_halo(MM, zz, kk)
        bb = self.bias(MM, zz)
        dndm = self.dndm_z(MM, zz)
        #Units: Mpc/h^2      (Mpc/h)-3
        return y3d * bb * dndm

    def tsz_yy_ll(self,kk, l):
        """Window function for the tSZ power spectrum. Eq. 18 of Taburet 2010"""
        integrated = scipy.integrate.quad(self._tsz_integrand,0.1,1, (kk,l))
        #units:     Mpc^-2              Mpc
        return -2 * integrated * self.light / self.H0

    def _tsz_integrand(self, aa, kk, l):
        """Integrand for the tSZ calculation above.
        Dimensionless."""
        #Comoving distance to this scale factor.
        zz = 1./aa - 1
        #Mpc ^-1
        Tmass = self.tsz_mass_integral(zz, kk) * self.hubble
        rr = self.light / self.H0 * self.conformal_time(zz)
        Dplus = self.lingrowthfac(zz)
        return aa**(-2) * Tmass * Dplus * scipy.special.sph_jn(l, kk * rr) / hzoverh0(aa, self.omegam0)

    def tsz_window_function_limber(self, aa, kk):
        """The window function for the tSZ that appears in the C_l if we use the Limber approximation.
        This gets rid of the spherical bessels."""
        zz = 1./aa - 1
        Tmass = self.tsz_mass_integral(zz, kk) * self.hubble
        Dplus = self.lingrowthfac(zz)
        return -2 * self.light / self.H0 * aa**(-2) * Tmass * Dplus / hzoverh0(aa, self.omegam0)

    def isw_window_function_limber(self, aa, kk):
        """The window function for the ISW that appears in the C_l if we use the Limber approximation."""
        #d/da (D+/a) = 5/2 omega_M / (H^2 a^4) + D+ /a ( H'/H - 1 /a)
        #Zero if H  = omega_M a^-3/2, D+ ~ a as H'/H ~ -3/2 / a
        H2 = hzoverh0(aa, self.omegam0)**2
        Hprime = 1.5 * self.omegam0 / aa**4 / H2
        Dplusda = 2.5 * self.omegam0 / (H2* aa**4) - self.lingrowthfac(aa) / aa * ( Hprime + 1/aa)
        return 3 * self.H0**2 / self.light**2 * self.omegam0 / kk**2 * Dplusda

    def _crosscorr_integrand(self, aa, ll, func1, func2):
        """Compute the cross-correlation of the ISW and tSZ effect using the limber approximation."""
        zz = 1./aa - 1
        rr = self.light / self.H0 * self.conformal_time(zz)
        #This is the Limber approximation
        kk = (ll + 1/2) / rr
        return func1(aa, kk) * func2(aa,kk) / rr**2 * self.overden.PofK(kk)

    def crosscorr(self, ll, func1, func2):
        """Compute the cross-correlation of the ISW and tSZ effect using the limber approximation."""
        (cll, err) = scipy.integrate.quad(self._crosscorr_integrand, 0.333, 1, (ll, func1, func2))
        if err / (cll+0.01) > 0.1:
            raise RuntimeError("Err in C_l computation: ",err)
        return (ll*(ll+1))/2/math.pi*cll

