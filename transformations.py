import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates,gaussian_filter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import elasticdeform

def generate_neighbors(X_train):
    from sklearn.neighbors import NearestNeighbors

    # Fit a nearest neighbors model to the training data
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X_train)

    # Find the nearest neighbors of each data point in the training set
    distances, indices = nbrs.kneighbors(X_train)

    # Generate synthetic examples based on the nearest neighbors
    synthetic_examples = []
    for i in range(len(X_train)):
        nearest_neighbor_indices = indices[i]
        nearest_neighbors = X_train[nearest_neighbor_indices]
        synthetic_example = np.mean(nearest_neighbors, axis=0)
        synthetic_examples.append(synthetic_example)
    synthetic_examples = np.vstack(synthetic_examples)
    return synthetic_examples


def shift_spectrum(spectrum_array, shift_range, direction='right'):
    """
    Modified version with different shift per row, to fasten the process
    """

    num_samples, num_bands = spectrum_array.shape
    shifted_spectrum = np.zeros_like(spectrum_array)

    shift_amount = np.random.randint(1, shift_range+1)
    if direction == 'right':
        shifted_spectrum[:, shift_amount:] = spectrum_array[:,
                                                            :num_bands - shift_amount]
        # Extrapolate values from the beginning of the spectrum
        x = np.arange(shift_amount, num_bands)
        y = spectrum_array[:, shift_amount:]
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        shifted_spectrum[:, :shift_amount] = f(np.arange(shift_amount))

    elif direction == 'left':
        shifted_spectrum[:, :num_bands -
                            shift_amount] = spectrum_array[:, shift_amount:]
        # Extrapolate values from the end of the spectrum
        x = np.arange(0, num_bands - shift_amount)
        y = spectrum_array[:, :num_bands - shift_amount]
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        shifted_spectrum[:, -shift_amount:] = f(
            np.arange(num_bands - shift_amount, num_bands))
    return shifted_spectrum


def hapke(sig, angle):
    # First, regress the input spectra to obtain the albedo
    mu = np.cos(np.deg2rad(30))
    mu0 = np.cos(np.deg2rad(30))

    temp = (((mu0+mu)**2*(sig**2) + (1+4*mu0*mu*sig)*(1-sig))
            ** 0.5 - (mu0+mu)*sig) / (1+4*mu0*mu*sig)
    w = -(temp**2-1)

    # generate signatures for different angles
    sampledSigs = []
    mu_b = np.cos(np.deg2rad(angle))
    mu0_b = np.cos(np.deg2rad(angle))

    x2 = w / ((1+2*mu_b*np.sqrt(1-w))*(1+2*mu0_b*np.sqrt(1-w)))
    sampledSigs.append(x2)

    return np.column_stack(sampledSigs)


def libGeneratorHapke(inputSignature):
    # Generate signatures for each row of inputSignature with a random value of angle
    outputSignature = []
    for i in range(inputSignature.shape[0]):
        angle = np.random.uniform(0, 85)
        sig = inputSignature[i]
        sigs = hapke(sig, angle)
        outputSignature.append(sigs)

    return np.concatenate(outputSignature, axis=0).reshape(inputSignature.shape)

# Atmospheric model
from scipy.interpolate import interp1d
import scipy.io

def libGeneratorSimpleAtmospheric(m0_sample, desiredWavelengths=None):

    # load spectral profile of direct and diffuse illumination sunlilght
    atmosIlluminationModel = scipy.io.loadmat(
        r'<YOUR_PATH_HERE>\atmosIlluminationModel.mat')
    wavelength_nm = atmosIlluminationModel['wavelength_nm'].reshape(-1,)
    Esun_direct_light = atmosIlluminationModel['Esun_direct_light'].reshape(
        -1,)
    Esky_diffuse_light = atmosIlluminationModel['Esky_diffuse_light'].reshape(
        -1,)

    if desiredWavelengths is not None:
        # Interpolate it to aviris bands
        f_direct_light = interp1d(wavelength_nm, Esun_direct_light,
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
        f_diffuse_light = interp1d(wavelength_nm, Esky_diffuse_light,
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
        Esun_direct_light = f_direct_light(desiredWavelengths).reshape(-1,)
        Esky_diffuse_light = f_diffuse_light(desiredWavelengths).reshape(-1,)

    L = m0_sample.shape[0]
    M_all = []
    aalpha0 = 0

    for i in range(L):
        # sample random angle for each row of m0_sample
        angle = np.random.uniform(0, 85)
        # 0.6*np.random.rand(); % 0.05 *(pi/2) * (rand-0.5) + 0;
        aalpha_obs = 0.9*(np.pi/2)*np.random.rand()
        M_all.append(m0_sample[i]*((Esun_direct_light*np.cos(np.deg2rad(angle))+Esky_diffuse_light)/(
            0.00005+Esun_direct_light*np.cos(aalpha0)+Esky_diffuse_light)))
    M_all = np.vstack(M_all)
    return M_all

def spectral_flip(spectra):
    """
    Flips the spectrum of each row in the input array. Assumes each row is real and even length.

    Args:
        spectra: A 2D NumPy array representing the input spectra.

    Returns:
        A 2D NumPy array representing the flipped spectra of the input signals.
    """

    # Reverse the order of the frequency bins for each row
    flipped_ffts = np.flip(spectra, axis=1)

    return flipped_ffts



def elastic_distortion(spectra, alpha_r, sigma_r):
    distorted_spectra = np.empty_like(spectra)

    for i in range(spectra.shape[0]):
        spectrum = spectra[i]
        alpha = np.random.randint(1, alpha_r)
        sigma = np.random.randint(1, sigma_r)

        # Handle 1D spectra
        if spectrum.ndim == 1:
            spectrum = spectrum[np.newaxis, :]  # Convert to 2D array

        # Generate random displacement fields
        shape = spectrum.shape
        dx = gaussian_filter1d(
            np.random.uniform(-1, 1, size=shape[1]), sigma, mode='reflect') * alpha

        # Create grid of coordinates
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y, (-1, 1)), np.reshape(x+dx, (-1, 1))

        # Apply distortion using spline interpolation
        distorted_spectrum = map_coordinates(
            spectrum, indices, order=3, mode='reflect')
        distorted_spectrum = np.reshape(distorted_spectrum, shape)

        distorted_spectra[i] = distorted_spectrum

    return distorted_spectra

def random_permutation(spectrum_array):
    # Get the number of bands in each spectrum
    num_bands = spectrum_array.shape[1]
    
    # Generate a random permutation of the band indices
    band_indices = np.random.permutation(num_bands)
    
    # Group the adjacent bands together
    groups = []
    current_group = []
    for i in range(num_bands):
        current_group.append(band_indices[i])
        if i == num_bands - 1 or band_indices[i+1] != band_indices[i] + 1:
            groups.append(current_group)
            current_group = []
    
    # Permute the bands within each group
    for group in groups:
        np.random.shuffle(group)
    
    # Create a new spectrum array with the permuted bands
    permuted_spectrum_array = np.empty_like(spectrum_array)
    for i in range(spectrum_array.shape[0]):
        permuted_spectrum_array[i,:] = spectrum_array[i,band_indices]
    
    return permuted_spectrum_array


def group_random_permutation(hyperspectral_array, group_size):
    # Get the shape of the hyperspectral array
    num_rows, num_columns = hyperspectral_array.shape
    # Calculate the number of groups
    num_groups = num_columns // group_size
    
    # Create an array to store the permuted hyperspectral data
    permuted_array = np.zeros((num_rows, num_columns), dtype=object)
    
    # Apply random permutations to each row
    groups = np.array_split(np.arange(num_columns), num_groups)
    permuted_groups =np.random.permutation(groups).tolist()
    permuted_row = [item for sublist in permuted_groups for item in sublist]

        
    permuted_array = hyperspectral_array[:,permuted_row]
    return permuted_array



def random_erasure(spectra, p=0.1):
    """
    Modified version to fasten process
    """
    n_samples, n_features = spectra.shape
    # Copy the input array
    erased_spectra = spectra.copy()
    
    # Compute the width of the band to erase
    band_width = int(n_features * p)
    
    # Loop over each spectrum and erase a band of values
    start_idx = np.random.randint(0, n_features - band_width)
    end_idx = start_idx + band_width
    erased_spectra[:, start_idx:end_idx] = 0
        
    return erased_spectra



def spatial_transformation(patch,transform, params):
    # spatial_transformation(patch,transform, rotation_range=360, elastic_alpha=0.05, elastic_sigma=0.01)
    """
    Apply spatial transformations to an image patch.

    Args:
        patch (np.ndarray): Input image patch with shape (L, W, H).
        rotation_angle (float): Rotation angle in degrees.
        elastic_alpha (int): Alpha parameter for elastic deformation.
        elastic_sigma (int): Sigma parameter for elastic deformation.

    Returns:
        np.ndarray: Transformed image patch.
    """
    # Apply rotation
    L, W, H = patch.shape
    rotation_angle = np.random.randint(1,params['rotation_range']+1)
    if transform=='rotation':
        # Rotate the image using OpenCV
        rotated_patch = np.zeros((L, W, H), dtype=patch.dtype)
        center = (W // 2, H // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)

        for band in range(L):
            # Rotate each band individually
            rotated_patch[band] = cv2.warpAffine(patch[band], rotation_matrix, (W, H))
        deformed_patch = rotated_patch

    # Apply elastic deformation
    if transform=='elastic':
        L, W, H = patch.shape
        # Generate a random elastic deformation grid for all bands
        displacement_field = np.random.randn(W, H) * params['elastic_alpha']
    
        # Create a 3D mesh grid (L, W, H) representing all bands
        x, y, z = np.meshgrid(np.arange(L), np.arange(W), np.arange(H), indexing='ij')

        # Apply displacement field using Gaussian filter for all bands simultaneously
        deformed_patch = patch + gaussian_filter(displacement_field, sigma=params['elastic_sigma'])

    if transform=='flip':
        L, W, H = patch.shape

        # Create an empty array to store the mirrored hyperspectral data
        deformed_patch = np.empty((L, W, H), dtype=patch.dtype)

        # Mirror each band individually
        for band in range(L):
            deformed_patch[band, :, :] = np.flipud(patch[band, :, :])

    
    if transform=='displace':
        L, W, H = patch.shape

        # Create a random permutation for the spatial dimensions (W, H)
        permuted_indices = np.random.permutation(W * H)

        # Create a reshaping index to apply the permutation to the patch
        reshaping_index = np.unravel_index(permuted_indices, (W, H))

        # Apply the random displacement to the patch
        deformed_patch = patch[:, reshaping_index[0], reshaping_index[1]].reshape(L,W,H)


    if transform=='translation':
        L, W, H = patch.shape
        # Define the range of translation values for W and H dimensions
        min_translation = -5  
        max_translation = 5 

        # Generate random translation values within the specified range
        random_translation_W = random.randint(min_translation, max_translation)
        random_translation_H = random.randint(min_translation, max_translation)

        # Create the translation vector
        translation_vector = (random_translation_W, random_translation_H)

        # Apply the translation to the patch
        deformed_patch  = np.roll(patch, shift=translation_vector, axis=(1, 2))

    return deformed_patch 
