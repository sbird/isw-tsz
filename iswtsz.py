"""Module to implement the ISW and tSZ amplitudes from Cyril's mathematica notebook"""

import math
import numpy as np
import scipy
import scipy.integrate
import scipy.special
import emcee #hammer
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import halo_mass_function as hm
import concentration

class LinearGrowth(object):
    """Class to store functions related to the linear growth factor and the Hubble flow."""
    def __init__(self, omegam0=0.27, hub=0.7, w0=-1., wa=0.):
        self.omegam0 = omegam0
        self.hub = hub
        self.w0 = w0
        self.wa = wa
        self._aaa = np.linspace(0.05,1,60)
        self._growths = np.array([self._generate_lingrowthfac(aa)/aa for aa in self._aaa])
        #Normalise to z=0.
        self._lingrowthfac=scipy.interpolate.InterpolatedUnivariateSpline(self._aaa,self._growths)
        self._conformal = np.array([self._generate_conformal_time(aa) for aa in self._aaa])
        self.conformal_time = scipy.interpolate.InterpolatedUnivariateSpline(self._aaa,self._conformal)

    def hzoverh0(self, a):
        """ returns: H(a) / H0  = [omegam/a**3 + (1-omegam)]**0.5 """
        return np.sqrt(self.h2overh0(a))

    def h2overh0(self, a):
        """ returns: H^2(a) / H0^2  = [omegam/a**3 + (1-omegam)]"""
        #Include correction for w0,wa models. Note the a dependence disappears if w0, wa =-1,0
        return self.omegam0/a**3 + (1.-self.omegam0)*a**(-3*(1+self.w0+self.wa))*np.exp(-3*self.wa*(1-a))

    def _lingrowthintegrand(self, a):
        """ (e.g. eq. 8 in lukic et al. 2008)   returns: da / [a*H(a)/H0]**3 """
        return np.power(a * self.hzoverh0(a),-3)

    def _ang_diam_integrand(self, aa):
        """Integrand for the ang. diam distance: 1/a^2 H(a)"""
        return 1./(aa**2*self.hzoverh0(aa))

    def _conf_time_integrand(self, aa):
        """Integrand for the ang. diam distance: 1/a^2 H(a)"""
        return 1./self.hzoverh0(aa)

    def _generate_conformal_time(self, aa):
        """Conformal time to scale factor a: int(1/H da).
        This is dimensionless; to get dimensionful value divide by H0."""
        conf, err = scipy.integrate.quad(self._conf_time_integrand, aa, 1.)
        if err/(conf+0.01) > 0.1:
            raise RuntimeError("Err in angular diameter: ",err)
        return conf

    def angular_diameter(self, aa):
        """The angular diameter distance to some redshift in units of comoving Mpc."""
        #speed of light in km/s
        light = 2.99e5
        #Dimensionless comoving distance
        comoving,err = scipy.integrate.quad(self._ang_diam_integrand, aa, 1)
        if err/(comoving+0.01) > 0.1:
            raise RuntimeError("Err in angular diameter: ",err)
        #angular diameter distance
        H0 = self.hub * 100 # km/s/Mpc
        return comoving * light * aa / H0

    def _generate_lingrowthfac(self, aa):
        """
        returns: linear growth factor, b(a) normalized to 1 at z=0, good for flat lambda only
        a = 1/1+z
        b(a) = Delta(a) / Delta(a=1)   [ so that b(z=0) = 1 ]
        (and b(a) [Einstein de Sitter, omegam=1] = a)

        Delta(a) = 5 omegam / 2 H(a) / H(0) * integral[0:a] [da / [a H(a) H0]**3]
        equation  from  peebles 1980 (or e.g. eq. 8 in lukic et al. 2008) """
        ## 1st calc. for z=z
        lingrowth,err = scipy.integrate.quad(self._lingrowthintegrand,0.,aa)
        if err/lingrowth > 0.1:
            raise RuntimeError("Err in linear growth: ",err)
        lingrowth *= 5./2. * self.omegam0 * self.hzoverh0(aa)
        return lingrowth

    def lingrowthfac(self, aa):
        """Growth function, using the interpolator where we can, otherwise doing it for real."""
        growth = self._lingrowthfac(aa)*aa
        #if aa > self._aaa[0]:
            #growth = self._lingrowthfac(aa)*aa
        #else:
            #growth = self._generate_lingrowthfac(aa)
        return growth / self._lingrowthfac(1.)

    def EprimebyE(self, aa):
        """Returns d/da(H^2/H0^2) / (H^2/H0^2)"""
        wapower = -3*(1+self.w0+self.wa)
        Eprime = -1.5 * self.omegam0 / aa**4 + (1.-self.omegam0)*np.exp(-3*self.wa*(1-aa))*(wapower*aa**(wapower-1)+3*self.wa)
        return Eprime / self.h2overh0(aa)

    def Dplusda(self, aa):
        """The derivative of the growth function divided by a w.r.t a."""
        #This is d/da(H/H0) / (H/H0)
        #This is d D+/da
        Dplusda = 2.5 * self.omegam0 / aa**3 / self.h2overh0(aa)/self._lingrowthfac(1.)
        Dplusda += self.EprimebyE(aa) * self.lingrowthfac(aa)
        return Dplusda/ aa - self.lingrowthfac(aa) /aa**2

class YYgas(object):
    """Helper class for the gas profile, which allows us to precompute the prefactors.
    Expects units without the h!"""
    def __init__(self, conc, mass, rhogas_cos, R200, tszbias=1.):
        self.gamma = self._gamma(conc)
        self.eta0 = self._eta0(conc)
        self.BB = 3./self.eta0 * (self.gamma - 1)/self.gamma / (np.log(1+conc)/conc - 1/(1+conc))
        #Newtons constant in units of m^3 kg^-1 s^-2
        gravity = 6.67408e-11
        #Boltzmann constant in m2 kg s-2 K-1
        kboltz = 1.38e-23
        #Solar mass in kg
        solarmass = 1.98855e30
        #Proton mass in kg
        protonmass = 1.6726219e-27
        #1 Mpc in m
        Mpc = 3.086e22
        #In M_sun / Mpc^3, as the constant below.
        #Eq. 21
        self.rhogas0 = rhogas_cos * conc**2 / (np.log(1+conc) - conc/(1+conc)) /((1+conc)**2*self.ygas(conc))
        # Units                                    m^3 s^-2                      kg                m        K s^2 / m^2 /kg
        self.Tgas0 = self.eta0 * 4 / (3+5*0.76) * gravity * protonmass * (mass * solarmass) /3/(R200* Mpc)/kboltz
        self.Pgas0 =  (3 + 5 * 0.76) / 4 * self.rhogas0 * solarmass / (protonmass) * kboltz * self.Tgas0 / Mpc**2
        #Electron mass in kg
        me = 9.11e-31
        #Thompson cross-section in m^2
        sigmat = 6.6524e-29
        #speed of light in m/s
        light = 2.99e8
        #Units are s^2 / kg
        self.y3d_prefac = tszbias * sigmat / me / light**2

    def _gamma(self, conc):
        """Polytropic index for a cluster from K&S 02, eq. 17"""
        return 1.137 + 8.94e-2 * np.log(conc/5) - 3.68e-3 * (conc-5)

    def _eta0(self, conc):
        """Mass-temperature normalisation factor at the center, eta(0)."""
        return 2.235 + 0.202 * (conc - 5) - 1.16e-3*(conc-5)**2

    def logxbyx(self, x):
        """Helper function with a series expansion for small x"""
        if x < 1e-4:
            ll1p = (x/2. - x**2/3 + x**3/4 - x**4/5)
        else:
            ll1p = (1-np.log1p(x)/x)
        return ll1p

    def ygas(self, x):
        """Dimensionless part of the K&S pressure profile, eq. 15"""
        #K&S 02 eq. 17 and 18, fitting formulae.
        #Series expansion for small x
        try:
            ll1p = np.array([self.logxbyx(z) for z in x])
        except TypeError:
            ll1p = self.logxbyx(x)
        return (1 - self.BB * ll1p)**(1./(self.gamma-1))

    def Pgas(self, x):
        """Gas pressure profile, K&S 02 eq 8.
        Other choices are possible (indeed preferred)!
        Units are kg / Mpc /s^2"""
        return self.Pgas0 * self.ygas(x)**self.gamma

    def y3d(self, x):
        """Electron pressure profile from K&S 2002 eq 7. Units of 1 / Mpc."""
        return (2 + 2*0.76)/(3 + 5*0.76) * self.Pgas(x) * self.y3d_prefac

class TSZHalo(object):
    """Extends the NFW halo calculation with equations for a tSZ profile."""
    def __init__(self, omegam0 = 0.27, omegab0=0.0449, hubble = 0.7, sigma8=0.8, tszbias=1., w0=-1., wa=0.):
        self.omegam0 = omegam0
        self.omegab0 = omegab0
        self.hubble = hubble
        self.tszbias=tszbias
        #km/s/Mpc
        self.H0 = 100* self.hubble
        #speed of light in km/s
        self.light = 2.99e5
        #tSZ frequency
        self.nu = 100.
        self.overden = hm.Overdensities(0,omegam0, omegab0,1-omegam0,hubble, 0.97,sigma8,log_mass_lim=(10,18))
        self.conc_model = concentration.LudlowConcentration(self.Dofz)
        self._rhocrit0 = self.overden.rhocrit(0)
        self.lingrowth = LinearGrowth(omegam0=omegam0, hub=hubble, w0=w0, wa=wa)

    def Dofz(self, zz):
        """Helper"""
        aa = 1./(1+zz)
        return self.lingrowth.lingrowthfac(aa)

    def mass_function(self, sigma):
        """Tinker et al. 2008, eqn 3, Delta=200 # Delta=200"""
        A = 0.186
        a = 1.47
        b = 2.57
        c = 1.19
        return A * ( np.power((sigma / b), -a) + 1) * np.exp(-1 * c / sigma / sigma)

    def dndm_z_b(self, mass, aa):
        """Returns the halo mass function dn/dM in units of h^4 M_sun^-1 Mpc^-3
        Requires mass in units of M_sun /h
        (times by a halo bias)"""
        # Mean matter density at redshift z in units of h^2 Msolar/Mpc^3
        # This is the comoving density at redshift z in units of h^-1 M_sun (Mpc/h)^-3 (comoving)
        # Compute as a function of the critical density at z=0.
        rhom = self.overden.rhocrit(0) * self.omegam0
        growth = self.lingrowth.lingrowthfac(aa)
        sigma = self.overden.sigmaof_M(mass)*growth
        mass_func = self.mass_function(sigma)
        dlogsigma = self.overden.log_sigmaof_M(mass)*growth/mass
        #We have dn / dM = - d ln sigma /dM rho_0/M f(sigma)
        dndM = - dlogsigma*mass_func/mass*rhom
        nu = 1.686/sigma
        return dndM * self.bias(nu)

    def concentration(self,mass, aa):
        """Compute the concentration for a halo mass"""
        zz = 1./aa-1
        nu = 1.686/self.overden.sigmaof_M(mass)
        return self.conc_model.comoving_concentration(nu, zz)

    def R200(self, mass):
        """Get the virial radius in comoving Mpc/h for a given mass in Msun/h"""
        #Units are Msun  h^2 / Mpc^3
        rhoc = self._rhocrit0
        #Virial radius R200 in Mpc/h from the virial mass
        return np.cbrt(3 * mass / (4* math.pi* 200 * rhoc))

    def tsz_per_halo(self, M, aa,kk):
        """The 2D fourier transform of the projected Compton y-parameter, per halo.
        Eq 2 of Komatsu and Seljak 2002, slightly modified to remove some of the dimensionless parameters.
        Takes mass in units of Msun/h, returns units of Mpc/h^2."""
        #Upper limit is the virial radius.
        conc = self.concentration(M, aa)
        #Get rid of factor of h
        R200 = self.R200(M)/self.hubble
        rhogas_cos = (self.omegab0  / self.omegam0) * 200 * self._rhocrit0
        ygas = YYgas(conc, M/self.hubble, rhogas_cos*self.hubble**2, R200, tszbias=self.tszbias)
        #Also no factor of h
        Rs = R200/conc
        redk = kk * Rs
        limit = 5/redk
        integrated,err = scipy.integrate.quad(self._ygas3d_integrand, 0, np.min([conc,limit]), (redk, ygas))
        if err/integrated > 0.1:
            raise RuntimeError("Err in tsz integral: ",err, integrated)
        #For large k the integrand becomes oscillatory and is exponentially suppressed.
        #So approximate it to save breaking the integrator.
        #This is computed by doing a contour integral and taking a (very bad) upper bound for the y_gas part.
        #Use a bad approximation as zero is actually not bad either.
        if conc > limit:
            integrated += ygas.y3d(limit) * (limit * np.exp(-limit * redk)-conc * np.exp(-conc * redk))/redk**2
        #Units:  Mpc*2
        return 4 * math.pi * Rs**3 * integrated*self.hubble**2

    def _ygas3d_integrand(self, xx, kk, ygas):
        """Integrand of the TSZ profile from Komatsu & Seljak 2002 eq. 2.
        Units are Mpc. Takes dimensionless r/Rs and dimensionless k = k * Rs"""
        #The bessel function does little unless k is large.
        integrand = ygas.y3d(xx) * xx * xx
        return scipy.special.j0(xx*kk) * integrand

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
        rr = self.light / self.H0 * self.lingrowth.conformal_time(aa)
        #d/da (D+/a) = 5/2 omega_M / (H^2 a^4) + D+ /a ( H'/H - 1 /a)
        #Zero if H  = omega_M a^-3/2, D+ ~ a as H'/H ~ -3/2 / a
        Dplusda = self.lingrowth.Dplusda(aa)
        return Dplusda * scipy.special.sph_jn(l, kk * rr)

    def bias(self,nu):
        """Formula for the bias of a halo, from Sheth-Tormen 1999."""
        delta_c = 1.686
        bhalo = (1 + (0.707*nu**2 -1)/ delta_c) + 2 * 0.3 * delta_c / (1 + (0.707 * nu**2)**0.3)
        return bhalo

    def tsz_mass_integral(self, aa, kk, mass=(1e12, 1e16)):
        """Formula for the mass integral of the tSZ power, Taburet 2010, 1012.5036.
        Units of h/Mpc"""
        (integrated, err) = scipy.integrate.quad(self._tsz_mass_integrand, np.log(mass[0]), np.log(mass[1]), (aa, kk))
        if err/(integrated+0.01) > 0.1:
            raise RuntimeError("Err in tsz mass integral: ",err," total:",integrated)
        return integrated

    def _tsz_mass_integrand(self, logM, aa, kk):
        """Integrand for the tSZ mass integral. Units of 1/Mpc"""
        MM = np.exp(logM)
        y3d = self.tsz_per_halo(MM, aa, kk/aa)
        dndm_b = self.dndm_z_b(MM, aa)
        #Units: Msun/h Mpc/h^2      (Mpc/h)-3/(Msun/h)
        return MM * y3d * dndm_b * self.hubble

    def tsz_yy_ll(self,kk, l):
        """Window function for the tSZ power spectrum. Eq. 18 of Taburet 2010"""
        integrated = scipy.integrate.quad(self._tsz_integrand,0.1,1, (kk,l))
        #units:     Mpc^-2              Mpc
        return self.gtsz(self.nu) * integrated * self.light / self.H0

    def _tsz_integrand(self, aa, kk, l):
        """Integrand for the tSZ calculation above.
        Dimensionless."""
        #Mpc ^-1
        Tmass = self.tsz_mass_integral(aa, kk)
        rr = self.light / self.H0 * self.lingrowth.conformal_time(aa)
        Dplus = self.lingrowth.lingrowthfac(aa)
        return aa**(-2) * Tmass * Dplus * scipy.special.sph_jn(l, kk * rr) / self.lingrowth.hzoverh0(aa)

    def tsz_window_function_limber(self, aa, kk):
        """The window function for the tSZ that appears in the C_l if we use the Limber approximation.
        This gets rid of the spherical bessels."""
        Tmass = self.tsz_mass_integral(aa, kk)
        Dplus = self.lingrowth.lingrowthfac(aa)
        return self.gtsz(self.nu) * self.light / self.H0 * aa**(-2) * Tmass * Dplus / self.lingrowth.hzoverh0(aa)

    def isw_window_function_limber(self, aa, kk):
        """The window function for the ISW that appears in the C_l if we use the Limber approximation."""
        #d/da (D+/a) = D+' / a - D+ / a^2
        #5/2 omega_M / (H^2 a^4) + D+ /a ( H'/H - 1 /a)
        #Zero if H  = omega_M a^-3/2, D+ ~ a as H'/H ~ -3/2 / a
        Dplusda = self.lingrowth.Dplusda(aa)
        #Units:                 Mpc^-2                          Mpc^2
        return 3 * self.H0**2 / self.light**2 * self.omegam0 / kk**2 * Dplusda * aa**2

    def _crosscorr_integrand(self, aa, lmode, func1, func2):
        """Compute the cross-correlation of the ISW and tSZ effect using the limber approximation."""
        #Units of rr are Mpc.
        rr = self.light / self.H0 * self.lingrowth.conformal_time(aa)
        #This is the Limber approximation: 1/Mpc
        kk = (lmode + 1/2) / rr
        #Convert k into h/Mpc and convert the result from (Mpc/h)**3 to Mpc**3
        PofKMpc = self.overden.PofK(kk/self.hubble) * self.hubble**3
        #Functions are dimensionless, so this needs to be dimensionless too.
        return func1(aa, kk) * func2(aa,kk) / rr**2 * PofKMpc * self.lingrowth.hzoverh0(aa) * self.H0 / self.light

    def kk_limber(self, lmode, aa):
        """Compute the k-mode value for a given l in the limber approximation."""
        rr = self.light / self.H0 * self.lingrowth.conformal_time(aa)
        #This is the Limber approximation
        kk = (lmode + 1/2) / (rr+1e-12)
        return kk

    def crosscorr(self, lmode, func1, func2):
        """Compute the cross-correlation of the ISW and tSZ effect using the limber approximation."""
        (cll, err) = scipy.integrate.quad(self._crosscorr_integrand, 0.333, 1, (lmode, func1, func2))
        if err / (cll+0.01) > 0.1:
            raise RuntimeError("Err in C_l computation: ",err)
        return (lmode*(lmode+1))/2/math.pi*cll

    def gtsz(self,nu):
        """Frequency dependent part of the tSZ power."""
        x = nu/56.78
        expx = math.exp(x)
        return x * (expx+1)/(expx-1) - 4

def lnprob(param, lvals, ISWtszdata, ISWtszCovar, ISWdata, ISWCovar, tszprior=0.8, tszwidth=0.25):
    """Likelihood function for emcee.
        The tSZ data is included as a prior on the mass bias of the tSZ halos
        param - vector of parameters.
        param[0] = tszbias (priored)
        param[1] = w0
        param[2] = wa
        rpar, rperp - r values for Cdata
        rr - r-values for Cpeak and Csmooth
        Cpeak - correlation function for the peak shape (mom, l, r)
        Csmooth - correlation function for the smooth shape (mom, l, r)
        where mom is integral xi_l (mu ^n * P L) for n = 0,1,2
        Cdata - observed correlation functions (rpar, rperp)
    """
    tszbias = param[0]
    w0 = param[1]
    wa = param[2]
    simulation = TSZHalo(tszbias=tszbias, w0=w0, wa=wa)
    #Compute the isw and tsz parameters
    iswxisw = np.array([simulation.crosscorr(l, simulation.isw_window_function_limber,simulation.isw_window_function_limber) for l in lvals])
    iswxtsz = np.array([simulation.crosscorr(l, simulation.isw_window_function_limber,simulation.tsz_window_function_limber) for l in lvals])
    #Interpolators for the smooth and peak correlation templates
    #Compute cosmological correlation for each r-value in the data
    chisq  = np.sum((iswxtsz - ISWtszdata)**2/ISWtszCovar)
    chisq += np.sum((iswxisw - ISWdata)**2/ISWCovar)
    chisq += np.sum((tszbias - tszprior)**2/tszwidth)
    return - np.log(chisq/2.)

if __name__ == "__main__":
    ll = np.linspace(4,500, 80)
    ttisw = TSZHalo()
    tsztsz = np.array([ttisw.crosscorr(l, ttisw.tsz_window_function_limber,ttisw.tsz_window_function_limber) for l in ll])
    plt.loglog(ll, tsztsz, label="tSZ")
    iswisw = np.array([ttisw.crosscorr(l, ttisw.isw_window_function_limber,ttisw.isw_window_function_limber) for l in ll])
    plt.loglog(ll, iswisw, label="ISW")
    iswtsz = np.array([ttisw.crosscorr(l, ttisw.isw_window_function_limber,ttisw.tsz_window_function_limber) for l in ll])
    plt.loglog(ll, iswtsz, label="ISW-tSZ")
    plt.legend()
    plt.savefig("ISWtsz.pdf")
    plt.clf()
    #Plot redshift dependence
    aaa = np.linspace(0.333, 1, 80)
    tszzz = np.array([ttisw.tsz_window_function_limber(a, ttisw.kk_limber(100, a)) for a in aaa])
    plt.plot(aaa, tszzz/np.min(tszzz), label="tSZ")
    iswzz = np.array([ttisw.isw_window_function_limber(a, ttisw.kk_limber(100, a)) for a in aaa])
    plt.plot(aaa, iswzz/np.min(iswzz), label="ISW")
    plt.legend()
    plt.savefig("redshift.pdf")
    plt.clf()

    if False:
        ndim, nwalkers = 3,1000
        p0 = [np.array([(2-0.4)*np.random.random()+0.4,(-1.2+0.7)*np.random.random()-0.7, (-1.-0.7)*np.random.random()+0.7]) for _ in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[ll, iswtsz, 0.1/iswtsz**2, iswisw, 0.1/iswisw**2, 1., 0.25])
        sampler.run_mcmc(p0, 1000)
