from classy import Class
import numpy as np
from itertools import combinations_with_replacement as cwr

def cl_to_dl_class(ell, cl):
    dl = ell * (ell + 1) * cl / (2 * np.pi)
    return(1e12 * dl)

def get_spectra(lmin, lmax, params):

    classSettings = params.copy()
    classSettings["output"] = "tCl,pCl,lCl,mPk"
    classSettings["lensing"] = "yes"
    classSettings["non linear"] = "HMcode"
    classSettings["P_k_max_h/Mpc"] = 100
    classSettings["z_max_pk"] = 2.
    classSettings["m_ncdm"] = 0.06
    classSettings["N_ncdm"] = 1
    classSettings["N_ur"] = 2.0328
    classSettings["l_max_scalars"] = lmax + 1000

    logA = classSettings.pop("logA")
    classSettings["A_s"] = 1e-10 * np.exp(logA)
    classBoltz = Class()
    classBoltz.set(classSettings)
    classBoltz.compute()

    classBoltz = classBoltz.lensed_cl()
    ell = classBoltz.get("ell")[lmin : lmax + 1]
    spec = {"ell": ell}

    for mode in ["tt", "te", "ee"]:
        cl = classBoltz.get(mode)[lmin : lmax + 1]
        spec[mode.upper()] = cl_to_dl_class(ell, cl)

        if mode == "te":
            spec["ET"] = cl_to_dl_class(ell, cl)

    return(spec)

def get_derivative(lmin, lmax, constrainedPars, **params):

    derivDict = {}
    for par in constrainedPars:

        derivDict[par] = {}
        paramLeft = params.copy()
        paramLeft[par] = 0.995 * paramLeft[par]

        paramRight = params.copy()
        paramRight[par] = 1.005 * paramRight[par]

        leftSpectra = get_spectra(lmin, lmax, paramLeft)
        rightSpectra = get_spectra(lmin, lmax, paramRight)

        derivDict[par]["ell"] = leftSpectra["ell"]
        for mode in ["TT", "TE", "EE"]:

            derivDict[par][mode] = (
                rightSpectra[mode] - leftSpectra[mode]
                                   ) / (paramRight[par] - paramLeft[par])

    return(derivDict)

def compute_covariance(dl_dict, nl_dict,
                       RS, XY, nsplits, fsky):

    R, S = RS
    X, Y = XY

    ell = dl_dict["ell"]
    pr = 1 / (2 * ell + 1) / fsky
    N = nsplits * (nsplits - 1)

    cov = pr * (dl_dict[R+X] * dl_dict[S+Y] +
                dl_dict[R+Y] * dl_dict[S+X])

    cov += pr / nsplits * (dl_dict[R+X] * nl_dict[S+Y] +
                           dl_dict[S+Y] * nl_dict[R+X])
    cov += pr / nsplits * (dl_dict[R+Y] * nl_dict[S+X] +
                           dl_dict[S+X] * nl_dict[R+Y])
    cov += pr / N * (nl_dict[R+X] * nl_dict[S+Y] +
                     nl_dict[R+Y] * nl_dict[S+X])

    return(np.diag(cov))

def get_covariance_dict(dl_dict, nl_dict, nsplits,
                        fsky, modes):

    ell = dl_dict["ell"]
    xmodes = cwr(modes, 2)
    covDict = {}

    for (m1, m2) in xmodes:

        covDict[m1+m2] = compute_covariance(dl_dict, nl_dict,
                                            m1, m2, nsplits,
                                            fsky)
    return(covDict)

def build_covariance(covDict, modes):

    covMat = []
    for i, m1 in enumerate(modes):
        line = []
        for j, m2 in enumerate(modes):

            if i <= j:
                line.append(covDict[m1+m2])
            elif i > j:
                line.append(covDict[m2+m1].T)
        covMat.append(line)

    covMat = np.block(covMat)
    return(covMat)

def build_fisher_mat(invCov, derivDl, modes):

    vectorDict = {}
    for par in derivDl:
        derivVect = []
        for mode in modes:
            derivVect = np.concatenate((derivVect, derivDl[par][mode]))
        vectorDict[par] = derivVect

    fisherDict = {}
    fisherMat = []

    for par1 in derivDl:
        line = []
        for par2 in derivDl:

            fp1p2 = vectorDict[par1] @ invCov @ vectorDict[par2]
            fisherDict[par1, par2] = fp1p2
            line.append(fp1p2)
        fisherMat.append(line)

    return(fisherDict, np.array(fisherMat))
