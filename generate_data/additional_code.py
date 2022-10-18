import numpy as np
from wave_propagation import spectral_del

def del2_iso9p(v, dx):
    """
    evaluate isotropic discrete Laplacian with 9-point stencil
    """

    Lv = (0.25 * np.roll(v, [1, 1], axis=[0, 1]) + 0.25 * np.roll(v, [-1, 1], axis=[0, 1])) + \
         (0.25 * np.roll(v, [1, -1], axis=[0, 1]) + 0.25 * np.roll(v, [-1, -1], axis=[0, 1])) + \
         (0.5 * np.roll(v, 1, axis=1) + 0.5 * np.roll(v, -1, axis=1)) + \
         (0.5 * np.roll(v, 1, axis=0) + 0.5 * np.roll(v, -1, axis=0)) - 3 * v

    return Lv / (dx ** 2)


def wave2_iso9p(u0, ut0, vel, dx, dt, Tf):
    """
    propagate wavefield using velocity Verlet in time and the isotropic Laplacian
    """
    Nt = round(abs(Tf / dt))
    c2 = np.multiply(vel, vel)

    u = u0
    ut = ut0

    for i in range(Nt):
        # Velocity Verlet
        ddxou = del2_iso9p(u, dx)
        u = u + dt * ut + 0.5 * dt ** 2 * np.multiply(c2, ddxou)
        ddxu = del2_iso9p(u, dx)
        ut = ut + 0.5 * dt * np.multiply(c2, ddxou + ddxu)

    return u, ut


def periLaplacian9p(v, dx):
    """
    evaluate discrete Laplacian with 9-point stencil
    """
    Lv = (-1. / 12 * np.roll(v, 2, axis=0) + 4. / 3. * np.roll(v, 1, axis=0)) + \
         (-1. / 12 * np.roll(v, -2, axis=0) + 4. / 3. * np.roll(v, -1, axis=0)) + \
         (-1. / 12 * np.roll(v, 2, axis=1) + 4. / 3. * np.roll(v, 1, axis=1)) + \
         (-1. / 12 * np.roll(v, -2, axis=1) + 4. / 3. * np.roll(v, -1, axis=1)) - 5. * v

    return Lv / (dx ** 2)


def wave2_9p(u0, ut0, vel, dx, dt, Tf):
    """
    propagate wavefield using velocity Verlet in time and the 9-point discrete Laplacian
    """

    Nt = round(abs(Tf / dt))
    c2 = np.multiply(vel, vel)
    u = u0
    ut = ut0

    for i in range(Nt):
        # Velocity Verlet
        ddxou = periLaplacian9p(u, dx)
        u = u + dt * ut + 0.5 * dt ** 2 * np.multiply(c2, ddxou)
        ddxu = periLaplacian9p(u, dx)
        ut = ut + 0.5 * dt * np.multiply(c2, ddxou + ddxu)

    return u, ut


def ftdt_pseudo_spectral(u0, ut0, vel, dx, dt, Tf):
    """
    propagate wavefield using velocity Verlet in time and spectral approx.
    of Laplacian in space
    """

    Nt = round(abs(Tf / dt))
    c2 = np.multiply(vel, vel)

    u = u0
    ut = ut0
    for i in range(Nt):
        # Velocity Verlet
        ddxou = spectral_del(u, dx)  # laplacian spectrally computed
        u = u + dt * ut + 0.5 * dt ** 2 * np.multiply(c2, ddxou)
        ddxu = spectral_del(u, dx)
        ut = ut + 0.5 * dt * np.multiply(c2, ddxou + ddxu)

    return np.real(u), np.real(ut)