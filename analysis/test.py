import torch

def velocity_verlet(x, v, a, dt, dx, pml_width):
  """
  Perform a single time step of the velocity Verlet algorithm for 2D wave propagation.
  x: current position of the wave (2D image)
  v: current velocity of the wave (2D image)
  a: acceleration of the wave (2D image)
  dt: time step
  dx: spatial step
  pml_width: width of the PML layer
  """
  # update position
  x += v*dt + 0.5*a*dt**2

  # calculate new acceleration
  a_new = calculate_acceleration(x, dx, v)

  # apply PML to acceleration
  a_new = apply_pml(a_new, pml_width)

  # update velocity
  v += 0.5*(a + a_new)*dt

  # update acceleration
  a = a_new

  return x, v, a

def calculate_acceleration(x, dx, v):
  """
  Calculate the acceleration of the wave at each point in the grid.
  """
  # create a second-order finite difference stencil for the Laplacian
  stencil = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

  # convolve the stencil with the wave to calculate the Laplacian
  laplacian = convolve(x, stencil, mode='same')

  # divide by dx**2 to get the acceleration
  a = -laplacian / dx**2

  # add in the velocity-dependent term
  a += c(x)*v

  return a

def apply_pml(a, pml_width, sigma_max=1, kappa_max=1, alpha_max=1):
  """
  Apply the PML boundary conditions to the acceleration.
  """
  # create arrays for the PML parameters
  sigma = torch.zeros_like(a)
  kappa = torch.zeros_like(a)
  alpha = torch.zeros_like(a)

  # set the PML parameters in the boundary regions
  sigma[:pml_width] = sigma_max * (pml_width - np.arange(pml_width)) / pml_width
  sigma[-pml_width:] = sigma_max * (np.arange(pml_width) + 1) / pml_width
  sigma[:, :pml_width] = sigma_max * (pml_width - np.arange(pml_width))[:, np.newaxis] / pml_width
  sigma[:, -pml_width:] = sigma_max * (np.arange(pml_width) + 1)[:, np.newaxis] / pml_width

  kappa = 1 + (kappa_max - 1) * sigma / sigma_max
  alpha = alpha_max * sigma / sigma_max

  # apply the PML boundary conditions to the acceleration
  a[:pml_width] = a[:pml_width] / kappa[:pml_width] - 2 * alpha[:pml_width] * v[:pml_width] / dx
  a[-pml_width:] = a[-pml_width:] / kappa[-pml_width:] - 2 * alpha[-pml_width:] * v[-pml_width:] / dx
  a[:, :pml_width] = a[:, :pml_width] / kappa[:, :pml_width] - 2 * alpha[:, :pml_width] * v[:, :pml_width] / dx
  a[:, -pml_width:] = a[:, -pml_width:] / kappa[:, -pml_width:] - 2 * alpha[:, -pml_width:] * v[:, -pml_width:] / dx

  return a

from scipy.signal import convolve2d

def convolve(x, kernel, mode='same'):
  """
  Convolve the input array with the given kernel.
  """
  return convolve2d(x, kernel, mode=mode)


if __name__ == '__main__':
    a()
