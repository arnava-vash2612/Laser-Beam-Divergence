import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================
# LASER PARAMETERS
# ============================
wavelength = 633e-9      # Laser wavelength in meters (e.g., 633 nm for red light)
w0 = 1e-3                # Beam waist (minimum spot size) in meters (1 mm)
z_R = np.pi * w0**2 / wavelength  # Rayleigh range: distance where beam area doubles

# ============================
# SIMULATION RANGE
# ============================
z_vals = np.linspace(0, 5 * z_R, 300)         # z-axis: from 0 to 5*Rayleigh range (in 300 steps)
x_vals = np.linspace(-5e-3, 5e-3, 400)        # x-axis: cross-section from -5 mm to +5 mm
Z, X = np.meshgrid(z_vals, x_vals)            # 2D grid for visualization (not directly used in this version)

# ============================
# COMPUTE INTENSITY PROFILE
# ============================
# Beam width at each z position
W_all = w0 * np.sqrt(1 + (z_vals / z_R)**2)

# Gaussian beam intensity for each z (looping through W_all)
# Each row corresponds to intensity profile at a given z
I_all = np.array([np.exp(-2 * (x_vals**2) / w**2) for w in W_all])
I_all = I_all.T  # Transpose so shape becomes (x, z) = (400, 300)

# ============================
# PLOTTING SETUP
# ============================
fig, ax = plt.subplots(figsize=(10, 4))

# Create an empty image with correct dimensions, set color map and axes
cax = ax.imshow(I_all * 0, extent=[0, 5 * z_R * 1000, -5, 5], aspect='auto', cmap='inferno', origin='lower')

# Labels and title
ax.set_title("Laser Beam Divergence Simulation (2D Gaussian Beam)")
ax.set_xlabel("Distance z (mm)")
ax.set_ylabel("Beam height x (mm)")
colorbar = fig.colorbar(cax, label="Normalized Intensity")

# ============================
# ANIMATION FUNCTION
# ============================
def animate(i):
    # Show intensity up to frame i
    intensity = I_all[:, :i+1]  # Take all x, and z from 0 to i
    pad_width = I_all.shape[1] - intensity.shape[1]  # Padding to keep constant shape

    # Pad zeros to the right if needed to fill full image width
    if pad_width > 0:
        intensity = np.pad(intensity, ((0, 0), (0, pad_width)))

    # Update the image data with new intensity
    cax.set_data(intensity)
    return [cax]

# ============================
# CREATE AND DISPLAY ANIMATION
# ============================
ani = animation.FuncAnimation(fig, animate, frames=I_all.shape[1], interval=30, blit=False)
plt.tight_layout()
plt.show()
