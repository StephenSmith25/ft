import numpy as np


import matplotlib.pylab as plt
import re

from lmfit import minimize, Parameters, fit_report

# generate synthetic data with noise
x = np.linspace(0, 100)
np.random.seed(10)
eps_data = np.random.normal(size=x.size, scale=0.2)
data = 7.5 * np.sin(x*0.22 + 2.5) * np.exp(-x*x*0.01) + eps_data

variables = [10.0, 0.2, 3.0, 0.007]


XYZ = np.column_stack((x, data, eps_data))
print(re.sub('[\[\]]', '', np.array2string(XYZ, separator=',')))


def residual(params, x, data, eps_data):
    amp = params['amp']
    phaseshift = params['phase']
    freq = params['frequency']
    decay = params['decay']
    model = amp * np.sin(x*freq + phaseshift) * np.exp(-x*x*decay)
    return (data-model) / eps_data


params = Parameters()
params.add('amp', value=10)
params.add('decay', value=0.007)
params.add('phase', value=0.2)
params.add('frequency', value=3.0)

out = minimize(residual, params, args=(x, data, eps_data))
print(fit_report(out))


fig = plt.figure()
plt.plot(x, data)
plt.show()


