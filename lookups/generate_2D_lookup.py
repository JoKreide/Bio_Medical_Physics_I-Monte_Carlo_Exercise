import numpy as np
import matplotlib.pyplot as plt

electron_mass = 0.511

def klein_nishina (e,theta):
    """
    Calculates the differential Klein-Nishina Cross-section for photon Energy E1 and scatter angle theta
    :param e: Photon Energy prior to Scattering in MeV
    :param theta: scattering angle
    :return: Differential Klein-Nishina Cross-section
    """
    alpha = e / electron_mass
    e2 = e / (1 + alpha * (1 - np.cos(theta)))
    eta = e2 / e
    return eta ** 2 * (eta + (eta ** -1) - (np.sin(theta) ** 2))

def inv_cdf (cdf, ps, thetas):
    return thetas[np.searchsorted(cdf, ps, side = "left")]

if __name__ == '__main__':
    e_points = 250
    theta_points = 5000
    p_count = 1000
    plotting = False
    file_path = "compton angles.csv"
    function = klein_nishina

    e_min = 0.0001
    e_max = 1
    steps = (e_max-e_min)/e_points
    es = np.arange(e_min, e_max+steps, steps)



    thetas = np.pi*np.arange(0, 1+1/theta_points, 1/theta_points)

    pdf = np.array([function(e, thetas) for e in es])
    pdf = np.array([values / np.sum(values) for values in pdf])

    cdfs = np.cumsum(pdf,axis = 1)
    cdfs = np.array([cdf - np.min(cdf) for cdf in cdfs])
    cdfs = np.array([cdf/np.max(cdf) for cdf in cdfs])

    ps = np.arange(0,1+1/p_count,1/p_count)
    inv_cdfs = np.array([inv_cdf(cdf, ps, thetas) for cdf in cdfs])

    tot = np.vstack((ps, inv_cdfs))
    tot = np.vstack((np.insert(es,0,0),tot.T)).astype(float).T

    np.savetxt(file_path, tot.T, delimiter = ',')

    if(plotting):
        plt.figure("PDFs")
        plt.plot(thetas, pdf.T)
        plt.figure("Polar PDFs")
        plt.polar(thetas - np.min(thetas), pdf.T)
        plt.figure("CDFs")
        plt.plot(thetas, cdfs.T)
        plt.figure("inverse CDFs")
        plt.plot(ps, inv_cdfs.T)
        plt.show()




