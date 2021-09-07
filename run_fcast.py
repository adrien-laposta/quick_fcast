import yaml
import fcast_utils
import sys
import numpy as np
import os
import pickle

yaml_file = sys.argv[-1]

############################
##### Parse .yaml file #####
############################
with open(yaml_file) as yml:
    info = yaml.load(yml, Loader = yaml.FullLoader)

params = info["params"]

modes = info["modes"]

lmin = info["lmin"]
lmax = info["lmax"]

nsplits = info["nsplits"]
fsky = info["fsky"]
constrainedPars = info["pars_to_constrain"]

noise_T_file = info["noise_T_file"]
noise_E_file = info["noise_E_file"]

output_dir = info["output_file"]
############################
############################

print("Computing reference spectra ...")
Dl_dict = fcast_utils.get_spectra(lmin, lmax, params)


######################
##### Load noise #####
######################
if noise_T_file is not None:

    ell_T, nl_T = np.loadtxt(noise_T_file, unpack = True)
    id = np.where((ell_T >= lmin) & (ell_T <= lmax))
    nl_T = nl_T[id]

else:

    nl_T = np.zeros_like(Dl_dict["TT"])

if noise_E_file is not None:

    ell_E, nl_E = np.loadtxt(noise_E_file, unpack = True)
    id = np.where((ell_E >= lmin) & (ell_E <= lmax))
    nl_E = nl_E[id]

else:

    nl_E = np.zeros_like(Dl_dict["EE"])

Nl_dict = {"TT": nl_T,
           "TE": np.zeros_like(Dl_dict["TE"]),
           "ET": np.zeros_like(Dl_dict["TE"]),
           "EE": nl_E}
print("Noise dict loaded.")
######################
######################


#######################
##### Compute cov #####
#######################
covDict =  fcast_utils.get_covariance_dict(Dl_dict, Nl_dict,
                                           nsplits,
                                           fsky, modes)

covMat = fcast_utils.build_covariance(covDict, modes)
invcovMat = np.linalg.inv(covMat)
print("Covariance computed.")
#######################
#######################

##########################
##### Get deriv dict #####
##########################
print("Starting to compute derivatives ...")
derivDict = fcast_utils.get_derivative(lmin, lmax, constrainedPars, **params)
print("Done !")
##########################
##########################

fisher = fcast_utils.build_fisher_mat(invcovMat, derivDict, modes)[1]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pickle.dump(Dl_dict, open(output_dir + "power_spectra.pkl", "wb"))
pickle.dump(Nl_dict, open(output_dir + "noise_ps.pkl", "wb"))
pickle.dump(derivDict, open(output_dir + "deriv_ps.pkl", "wb"))

np.savetxt(output_dir + "cov_mat.dat", covMat)
np.savetxt(output_dir + "fisher_mat.dat", fisher)
np.savetxt(output_dir + "parameter_cov.dat", np.linalg.inv(fisher))
