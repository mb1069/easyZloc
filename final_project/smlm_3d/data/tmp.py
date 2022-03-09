from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal

def gaussian3DPDFFunc(X, mu0, mu1, mu2, s00, s01, s02, s01, s11, s12, s02, s12, s22, A):
    mu = np.array([mu0, mu1, mu2])
    cov = np.array([
        [s00, s01, s02], 
        [s01, s11, s12], 
        [s02, s12, s22]
    ])
    res = multivariate_normal.pdf(X, mean=mu, cov=cov)
    res *= A
    return res


def fitGaussian(cube):
    cube = cube / cube.max()
    # prepare the data for curvefit
    X = []
    Y = []
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            for k in range(cube.shape[2]):
                X.append([i,j,k])
                Y.append(cube[i][j][k])
    
    coord_bounds = [0, 0, 0], [c//2 for c in cube.shape], list(cube.shape)
    scale_bounds = [0.1, 0.1, 0.1], [1, 1, 1], [10, 10, 10]
    
    cov_bounds = [0.1] * 9, [1] * 9, [10] * 9
    A_bounds = [0.95], [1], [1.05]
    
    low_bounds = coord_bounds[0] + scale_bounds[0] + cov_bounds[0] + A_bounds[0]
    high_bounds = coord_bounds[2] + scale_bounds[2] + cov_bounds[2] + A_bounds[2]
    
    p0 = coord_bounds[1] + scale_bounds[1] + cov_bounds[1] + A_bounds[1]
    
    popt, pcov = curve_fit(gaussian3DPDFFunc, X, Y, p0, bounds=(low_bounds, high_bounds), maxfev=1000)
    mu = [popt[0], popt[1], popt[2]]
    sigma = [[popt[3], 0, 0], [0, popt[4], 0], [0, 0, popt[5]]]
    A = popt[6]
    res = multivariate_normal.pdf(X, mean=mu, cov=sigma)
    return mu, sigma, A, res

print(psf.shape)
res = fitGaussian(psf)
peak_z = round(res[0][0])
print(peak_z)
plt.imshow(psf[peak_z])
plt.show()
for r in res:
    print(r)