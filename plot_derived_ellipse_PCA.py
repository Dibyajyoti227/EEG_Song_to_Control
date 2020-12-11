import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def get_cov_ellipse(cov, centre, nstd, **kwargs):

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height= 2 * nstd * np.sqrt(eigvals)
#    print ("Width: ",width)
#    print ("Height: ",height)
#    print ("Angle (degrees): ",np.degrees(theta))
#    print ("\n")

    degree_angle = np.degrees(theta)
    return width, height, degree_angle, Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)

def plot_derived_ellipse(data,band,Feature1, Feature2,ax):

    Silence, Song = 0,1
#    fig, ax = plt.subplots()
    labels, colours =['Silence', 'Song'], ['magenta', 'blue']
    width = []
    height = []
    deg_angle = []
    coords_x = 0.70
    coords_y = -0.3
    for CB in (Silence, Song):
        df = data[data['Song-status']==CB]

        x = df[df['Features']==Feature1]
        y = df[df['Features']==Feature2]
        x_mean = np.mean(x[band])
        y_mean = np.mean(y[band])
        cov = np.cov(x[band], y[band])
#        cov = np.array([df])
#        cov = np.cov(df['principal component 1'], df['principal component 2'])
        ax.scatter(x[band], y[band], color=colours[CB],
                   label=labels[CB], s=3)
#        ax.scatter(df['principal component 1'], df['principal component 2'], color=colours[CB],
#                   label=labels[CB], s=3)
        w, h, d_a, e = get_cov_ellipse(cov, (x_mean, y_mean), 3,
                            fc=colours[CB], alpha=0.4)
        w_text = 'Width:' + str(round(w,2))
        h_text = 'Height:' + str(round(h,2))
        d_a_text = 'Angle:' + str(round(d_a,2))
        whole_text = w_text +'\n'+ h_text +'\n'+ d_a_text +'\n'
        ax.text(coords_x,coords_y,whole_text,color = colours[CB])
        coords_y += 1.3
        width.append(w)
        height.append(h)
        deg_angle.append(d_a)
        ax.add_artist(e)

    ax.set_xlim(-0.5, 2)
    ax.set_ylim(-0.5, 2)
    ax.set_xlabel(Feature1+'-'+ band)
    ax.set_ylabel(Feature2+'-'+band)
    ax.legend(loc='upper left', scatterpoints=1)
#    plt.show()
    return width,height,deg_angle