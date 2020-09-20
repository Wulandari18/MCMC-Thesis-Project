import numpy as np
from scipy.special import kv
from scipy.special import iv
import scipy.integrate as integrate

#Baca data kecepatan rotasi

data = np.loadtxt ('ESO116-G012_rotmod.dat')

rad 		= data[:,0]
rad_km		= rad * 3.086e+16		#km
v_obs 		= data[:,1]			#km/s
v_obs_kpc	= v_obs * 3.24e-17		#kpc/s
err_v		= data[:,2]			#km/s
err_v_kpc	= err_v * 3.24e-17		#kpc/s
v_gasmodel	= data[:,3]			#km/s
v_gasmodel_kpc	= v_gasmodel * 3.24e-17		#kpc/s
v_diskmodel	= data[:,4]			#km/s
v_diskmodel_kpc	= v_diskmodel * 3.24e-17	#kpc/s
r_last		= rad[-1]

#parameter
Rd_kpc = 1.51			#kpc
Rd_c = Rd_kpc * 3.086e+21	#cm
G = 6.674e-08			#cgs
MHI = 1.083e+9			#Msun
Mg = 1.33 * MHI			#Msun
Mg_c = 1.33 * MHI * 1.988e+33	#g

#Parameter Tebakan
initparam1 	= 1.2				#M_L
initparam2	= 5.0e+7			#rho0
initparam3	= 2.0				#Rc
initparam4 	= 3.0				#r0
initparam5 	= 3.0				#B0
initparam6	= 5.0				#R0

delta_param1 	= 0.01
delta_param2	= 1e+5
delta_param3	= 0.01
delta_param4	= 0.01
delta_param5 	= 0.01
delta_param6	= 0.01

#Definisi fungsi gabungan

def VR(rad_x, v1_diskmodel_kpc, v1_gasmodel_kpc, param1, param2, param3, param4, param5, param6):
	G	= 4.52e-39	#kpc^3/Msun*s^2
	integral= integrate.quad(lambda x: ((4 * np.pi * param2 * (param3 ** 3) * (rad_x**2))/((rad_x + param3) * ((rad_x ** 2) + (param3 ** 2)))), 0, r_last)
	v_halo	= G * integral [0]/r_last

	Den = (Mg/(2*np.pi*(param4)**2)) * np.exp(-rad_x/(param4))	#Msun/kpc^3
	Den_cgs = (Den * 1.988e+33)/(3.086e+21)**3			#cgs

	Den_0 = (Mg/(2*np.pi*(param4)**2))				#Msun/kpc^3
	Den_g = Den_0 * 1.988e+33
	Den_0c = (Den_0 * 1.988e+33)/(3.086e+21)**3			#cgs

	B2 = (param5 / (1 + (rad_x / param6))) ** 2
	difB2 = -(2 * (param5 ** 2) * (param6 ** 2)) / (param6 + rad_x) ** 3

	v_mag_cm = (rad_x / (4 * np.pi *  Den_cgs)) * (((B2 / rad_x) + (0.5 * difB2)) * 1e-12)
	v_magplot = v_mag_cm ** 0.5					#cm
	v_magfinal = v_magplot * (3.24e-22)

	v_disk  = v1_diskmodel_kpc * param1

	v_rot	= ((v_disk ** 2) + (v1_gasmodel_kpc ** 2) + (v_halo) + (v_magfinal ** 2)) ** 0.5
	return v_rot

def log_prior(param):
	param1, param2, param3, param4, param5, param6 = param

	param1_test = param1 + np.random.normal(0.0, delta_param1, 1)
	param2_test = param2 + np.random.normal(0.0, delta_param2, 1)
	param3_test = param3 + np.random.normal(0.0, delta_param3, 1)
	param4_test = param4 + np.random.normal(0.0, delta_param4, 1)
	param5_test = param5 + np.random.normal(0.0, delta_param5, 1)
	param6_test = param6 + np.random.normal(0.0, delta_param6, 1)

	if 0.2 < param1_test < 3.0:
		param1 = param1_test
	if 1e+6 < param2_test < 5e+8:
		param2 = param2_test
	if 0.5 < param3_test < 15.0:
		param3 = param3_test
	if 0.2 < param4_test < 8.0:
		param4 = param4_test
	if 0.2 < param5_test < 30.0:
		param5 = param5_test
	if 0.2 < param6_test < 30.0:
		param6 = param6_test
	elif param1_test < 0.2 or param1_test > 3.0:
		while param1_test < 0.2 or param1_test > 3.0:
			param1_test = param1 + np.random.normal(0.0, delta_param1, 1)
		param1 = param1_test
	elif param2_test < 1e+6 or param2_test > 5e+8:
		while param2_test < 1e+6 or param2_test > 5e+8:
			param2_test = param2 + np.random.normal(0.0, delta_param2, 1)
		param2 = param2_test
	elif param3_test < 0.5 or param3_test > 15.0:
		while param3_test < 0.5 or param3_test > 15.0:
			param3_test = param3 + np.random.normal(0.0, delta_param3, 1)
		param3 = param3_test
	elif param4_test < 0.2 or param4_test > 8.0:
		while param4_test < 0.2 or param4_test > 8.0:
			param4_test = param4 + np.random.normal(0.0, delta_param4, 1)
		param4 = param4_test
	elif param5_test < 0.2 or param5_test > 30.0:
		while param5_test < 0.2 or param5_test > 30.0:
			param5_test = param5 + np.random.normal(0.0, delta_param5, 1)
		param5 = param5_test
	elif param6_test < 0.2 or param6_test > 30.0:
		while param6_test < 0.2 or param6_test > 30.0:
			param6_test = param6 + np.random.normal(0.0, delta_param6, 1)
		param6 = param6_test
	return param1, param2, param3, param4, param5, param6

def log_likelihood(rad, param1, param2, param3, param4, param5, param6):
	chidisk = 0.
	for i in range(len(rad)):
		chidisk = (chidisk) + ((-0.5) * (np.log(2 * np.pi * (err_v_kpc[i] ** 2)) + ((v_obs_kpc[i] - VR(rad[i], v_diskmodel_kpc[i], v_gasmodel_kpc[i], param1, param2, param3, param4, param5, param6)) ** 2)/err_v_kpc[i] ** 2))
	return chidisk

def chi_square(rad, param1, param2, param3, param4, param5, param6):
	chi_sqdisk = 0.
	for i in range(len(rad)):
		obsdisk =((v_obs_kpc[i] - VR(rad[i], v_diskmodel_kpc[i], v_gasmodel_kpc[i], param1, param2, param3, param4, param5, param6))/(err_v_kpc[i])) ** 2
		chi_sqdisk = (chi_sqdisk + obsdisk)
	return chi_sqdisk

# METROPOLIS-HASTINGS MCMC
# ========================

n_iter = 800000
burn_in = int(0.8 * n_iter)

param1 = []
param2 = []
param3 = []
param4 = []
param5 = []
param6 = []

step = []
Like = []

for i in range (n_iter):
	param = initparam1, initparam2, initparam3, initparam4, initparam5, initparam6
	newparam = log_prior (param)
	likelihood2 = log_likelihood(rad, newparam[0], newparam[1], newparam[2], newparam[3], newparam[4], newparam[5])
	likelihood1 = log_likelihood(rad, initparam1, initparam2, initparam3, initparam4, initparam5, initparam6)

	q = likelihood2 - likelihood1
	if q >= np.log(1.0) :
		#print ('TERIMA-1')

		initparam1 = newparam[0]
		initparam2 = newparam[1]
		initparam3 = newparam[2]
		initparam4 = newparam[3]
		initparam5 = newparam[4]
		initparam6 = newparam[5]

		Like.append(likelihood2)

		param1.append(newparam[0])
		param2.append(newparam[1])
		param3.append(newparam[2])
		param4.append(newparam[3])
		param5.append(newparam[4])
		param6.append(newparam[5])

	elif q < np.log(1.0) :
		r1 = np.random.uniform(1e-6, 1.0, 1)
		r = np.log(r1)
		if q >= r :
			#print ('TERIMA-2')

			initparam1 = newparam[0]
			initparam2 = newparam[1]
			initparam3 = newparam[2]
			initparam4 = newparam[3]
			initparam5 = newparam[4]
			initparam6 = newparam[5]

			Like.append(likelihood2)

			param1.append(newparam[0])
			param2.append(newparam[1])
			param3.append(newparam[2])
			param4.append(newparam[3])
			param5.append(newparam[4])
			param6.append(newparam[5])

		elif q < r :
			#print ('TOLAK')

			initparam1 = param[0]
			initparam2 = param[1]
			initparam3 = param[2]
			initparam4 = param[3]
			initparam5 = param[4]
			initparam6 = param[5]

			Like.append(likelihood1)

			param1.append(param[0])
			param2.append(param[1])
			param3.append(param[2])
			param4.append(param[3])
			param5.append(param[4])
			param6.append(param[5])

	step.append(i)

	if i%1000.0 == 0.0:
		np.savetxt('1MDMparams.txt', np.c_[step, param1, param2, param3, param4, param5, param6])
		print(step[i], Like[i])

