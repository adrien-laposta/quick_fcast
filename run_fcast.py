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

noise_file = info["noise_file"]

output_dir = info["output_file"]
############################
############################

print("Computing reference spectra ...")
Dl_dict = fcast_utils.get_spectra(lmin, lmax, params)


######################
##### Load noise #####
######################
Nl_dict = fcast_utils.get_noise_dict(noise_file, lmin, lmax)
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
