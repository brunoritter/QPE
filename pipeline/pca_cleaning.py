import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from astropy import table
from astropy.wcs import WCS
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import copy
import glob
from scipy.interpolate import interp1d
from scipy.optimize import nnls
import pywt
from scipy.optimize import minimize


class Masks:
    def _emission_masker(self, input_signal):
        # Máscara para remover linhas de emissão do espectro
        masked_signal = np.copy(input_signal)
        for line in self.emission_lines:
            mask = (self.corrected_wl >= (line - self.delta_lambda)) & (
                self.corrected_wl <= (line + self.delta_lambda)
            )
            masked_signal = np.where(mask, np.nan, masked_signal)

        return masked_signal

    def _mask_emission_lines(self, input_source):
        # Aplica a máscara de linhas de emissão a cada espectro na fonte
        emission_masked_source = np.zeros_like(input_source)
        for i in range(emission_masked_source.shape[0]):
            emission_masked_source[i] = self._emission_masker(input_source[i])

        return emission_masked_source

    def _absortion_masker(self, input_signal):
        # Máscara para remover janelas de absorção do espectro
        masked_signal = np.copy(input_signal)
        for window in self.absortion_windows:
            mask = (self.corrected_wl >= window[0]) & (self.corrected_wl <= window[1])
            masked_signal = np.where(mask, np.nan, masked_signal)

        return masked_signal

    def _mask_absortion_windows(self, input_source):
        # Aplica a máscara de janelas de absorção a cada espectro na fonte
        absortion_masked_source = np.zeros_like(input_source)
        for i in range(absortion_masked_source.shape[0]):
            absortion_masked_source[i] = self._absortion_masker(input_source[i])

        return absortion_masked_source

    def mask_signal(self, input_signal):
        # Combina as máscaras de emissão e absorção para criar um espectro 'limpo'
        emission_masked_source = self._mask_emission_lines(input_signal)
        absortion_masked_source = self._mask_absortion_windows(emission_masked_source)
        self.masked_source = absortion_masked_source


class SpectralPCA(Masks):
    # Inicialização da classe, carregando dados do arquivo FITS e calculando o comprimento de onda corrigido
    def __init__(
        self,
        filename,
        redshift,
        emission_lines,
        emission_line_window_size,
        absortion_windows,
    ):
        self.filename = filename
        self.redshift = redshift
        self.emission_lines = emission_lines
        self.delta_lambda = emission_line_window_size
        self.absortion_windows = absortion_windows

        with fits.open(filename) as hdulist:
            data = hdulist["SCI"].data  # Dados científicos
            mdf = table.Table(hdulist["MDF"].data)  # Metadados
        beam_mask = mdf["BEAM"] == -1
        source_mask = (mdf["BEAM"] == 1)[~beam_mask]

        self.input_signal = data[source_mask]

        wcs = WCS(fits.getheader(filename, ext=("sci", 1)), naxis=[1])
        self.wavelength = wcs.wcs_pix2world(np.arange(self.input_signal.shape[1]), 0)[0]
        self.corrected_wl = self.wavelength / (1 + self.redshift)

        self.mask_signal(self.input_signal)

    def pca_decompose(self, n_components=20):
        # Realiza a decomposição PCA no espectro 'limpo', mantendo um número definido de componentes principais
        self.pca = PCA(n_components=n_components)
        self.pca = PCA(n_components=n_components)
        transposed_signal = self.masked_source.copy().T
        transposed_signal[np.isnan(transposed_signal)] = 0.0
        self.eigen_spectra = self.pca.fit_transform(transposed_signal).T
        self.tomograms = getattr(self.pca, "components_") + getattr(self.pca, "mean_")

    def tomography(self, component):
        # Gera gráficos para visualização da tomografia PCA de um componente específico
        fig, ax = plt.subplots(
            nrows=3, ncols=1, figsize=(25, 8), constrained_layout=True
        )
        ax[0].plot(np.nansum(self.masked_source, axis=1))
        ax[0].set_title("Perfil de luminosidade do sinal")
        ax[1].plot(self.tomograms[component])
        ax[1].set_title(f"Tomograma da componente {component}")
        ax[1].axhline(0, ls="dashed", color="black")
        ax[2].plot(self.corrected_wl, self.eigen_spectra[component])
        ax[2].set_title(f"Autoespectro (autovetor) {component}")
        ax[2].axhline(0, ls="dashed", color="black")

    def luminosity_component_correlation(self, selection_criteria):
        # Analisa a correlação entre a luminosidade e os componentes do PCA
        Y = pd.DataFrame(np.nansum(self.masked_source, axis=1))
        X = pd.DataFrame(self.tomograms.T)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_sm = sm.add_constant(X_scaled)
        model_sm = sm.OLS(Y, X_scaled_sm).fit()

        model_summary = pd.DataFrame(model_sm.summary2().tables[1])
        model_summary.drop("const", inplace=True)
        model_summary["t"] = np.abs(model_summary["t"])
        model_summary["Coef."] = np.abs(model_summary["Coef."])

        self.ordered_features = model_summary[selection_criteria].sort_values(
            ascending=False
        )
        print(self.ordered_features)

    def subtract_bad_components(self, selection_threshold=None, bad_components=None):
        if bad_components is not None:
            # Se uma lista de componentes ruins foi fornecida diretamente
            components_to_remove = [int(x) for x in bad_components]
        elif selection_threshold is not None:
            # Remove os componentes do PCA considerados 'ruins' com base em um limiar de seleção
            components_to_remove = self.ordered_features[
                self.ordered_features < selection_threshold
            ].index
            components_to_remove = [int(x.strip("x")) - 1 for x in components_to_remove]
        else:
            raise ValueError(
                "Either selection_threshold or bad_components must be provided."
            )

        print(f"Removing components: {components_to_remove}")

        # Realiza a subtração dos componentes ruins
        bad_eigen_spectra = copy.deepcopy(self.eigen_spectra)
        for i in range(len(bad_eigen_spectra)):
            if i not in components_to_remove:
                bad_eigen_spectra[i] = 0
        bad_signal = np.dot(self.tomograms.T, bad_eigen_spectra)
        subtracted_source = copy.deepcopy(self.masked_source)
        subtracted_source -= bad_signal

        self.noise_subtracted_source = subtracted_source

    def zero_bad_components_and_reconstruct(self, bad_components=None):
        print(f"Zeroing components: {bad_components}")

        # Criar uma cópia dos autoespectros e zerar os componentes ruins
        good_eigen_spectra = copy.deepcopy(self.eigen_spectra)
        for i in range(len(good_eigen_spectra)):
            if i in bad_components:
                good_eigen_spectra[i] = np.zeros_like(good_eigen_spectra[i])

        # Reconstruir o sinal usando apenas os componentes bons
        reconstructed_signal = np.dot(self.tomograms.T, good_eigen_spectra)
        self.reconstructed_source = reconstructed_signal

    def StepSignalRemoval(self, component):
        return StepSignalRemoval(
            input_signal=self.noise_subtracted_source,
            component=component,
            eigen_spectra=self.eigen_spectra,
            tomograms=self.tomograms,
            corrected_wl=self.corrected_wl,
            emission_lines=self.emission_lines,
            delta_lambda=self.delta_lambda,
            absortion_windows=self.absortion_windows,
        )


class StepSignalRemoval(Masks):
    def __init__(
        self,
        input_signal,
        component,
        eigen_spectra,
        tomograms,
        corrected_wl,
        emission_lines,
        delta_lambda,
        absortion_windows,
    ):
        self.input_signal = input_signal
        self.component = component
        self.eigen_spectra = eigen_spectra
        self.tomograms = tomograms
        self.corrected_wl = corrected_wl
        self.emission_lines = emission_lines
        self.delta_lambda = delta_lambda
        self.absortion_windows = absortion_windows
        self._read_starlight_base()

        # self.mask_signal(emission_lines, delta_lambda, absortion_windows)

    def _read_starlight_base(self, filepath_pattern="STARLIGHTv04/BasesDir/*spec"):
        wlmin = self.corrected_wl.min()
        wlmax = self.corrected_wl.max()
        files = glob.glob(filepath_pattern)
        files.sort()

        starlight_base = []
        wl = np.loadtxt(files[0], unpack=True)[0]

        mask = (wl > wlmin) & (wl < wlmax)
        wl = wl[mask]
        for spectrum in files:
            starlight_base.append(np.loadtxt(spectrum, unpack=True)[1][mask])
        starlight_base = np.column_stack(starlight_base)
        self.starlight_base = starlight_base

    def _step_fit(self, signal, fft_filter, wavelet_filter, plot=False):
        """
        Fits a function to a spectra, aiming to capture "steps" in the continuum.
        The steps originate from instrumental artifacts.
        The procedure firt applies low pass filter to the signal, using a fast fourrier transform.
        The second step is a wavelet filter, using a Haar wavelet.

        Parameters:
        fft_filter (int): The number of high frequencies to be filtered out of the signal.
        wavelet_filter (float): The threshold used to filter out wavelet components.
        """

        # FFT low pass filter
        fft_signal = np.fft.fft(signal)
        resolution = self.corrected_wl[1] - self.corrected_wl[0]
        frequencies = np.fft.fftfreq(len(signal), resolution)
        step = np.max(frequencies) / len(frequencies[: len(frequencies) // 2])
        frequency_cutoff = np.max(frequencies) - (step * fft_filter)
        # Creates a mask with low pass filter
        low_pass_filter = np.abs(frequencies) < frequency_cutoff

        # Apply mask
        filtered_fft_signal = fft_signal * low_pass_filter
        # Apply inverse FFT to reconstruct filetered signal
        fft_filtered = np.fft.ifft(filtered_fft_signal)

        # Wavelet  filter
        coeffs = pywt.wavedec(np.real(fft_filtered), "db1")
        coeffs_thresh = [pywt.threshold(c, wavelet_filter, mode="hard") for c in coeffs]
        fft_wavelet_filtered = pywt.waverec(coeffs_thresh, "db1")

        # Make sure all signals have the same range
        min_length = min(
            len(self.corrected_wl), len(fft_filtered), len(fft_wavelet_filtered)
        )
        tomo_wavelength_truncated = self.corrected_wl[:min_length]
        fft_filtered_truncated = fft_filtered[:min_length]
        fft_wavelet_filtered_truncated = fft_wavelet_filtered[:min_length]

        if plot == True:
            plt.figure(figsize=(30, 5))
            plt.plot(
                tomo_wavelength_truncated, signal[:min_length], label="Original Signal"
            )
            plt.plot(
                tomo_wavelength_truncated,
                fft_filtered_truncated,
                label="FFT filtered Signal",
                linestyle="--",
            )
            plt.plot(
                tomo_wavelength_truncated,
                fft_wavelet_filtered_truncated,
                label="FFT + Haar filtered Signal",
                linestyle="--",
            )
            plt.legend()
            plt.show()

        return fft_wavelet_filtered_truncated

    # Apply step filtering
    def calculate_step_eigensignal(self, fft_filter, wavelet_filter):
        og_component = np.copy(self.eigen_spectra[self.component])
        og_mean = np.mean(og_component)
        og_std = np.std(og_component)
        normalized_component = (og_component - og_mean) / og_std
        normalized_step_signal = self._step_fit(
            normalized_component, fft_filter, wavelet_filter
        )
        step_eigensignal = (normalized_step_signal * og_std) + og_mean
        return step_eigensignal

    def subtract_step_signal(self, step_eigensignal):
        step_eigenspectra = copy.deepcopy(self.eigen_spectra)
        for i in range(len(step_eigenspectra)):
            if i != self.component:
                step_eigenspectra[i] = 0
        step_eigenspectra[self.component] = step_eigensignal.T

        self.bad_signal = np.dot(self.tomograms.T, step_eigenspectra)
        step_corrected_spectrum = copy.deepcopy(self.input_signal)
        step_corrected_spectrum = step_corrected_spectrum - self.bad_signal
        return step_corrected_spectrum

    def calculate_and_subtract_step_signal(self, fft_filter, wavelet_filter):
        step_signal = self.calculate_step_eigensignal(fft_filter, wavelet_filter)
        step_corrected_spectrum = self.subtract_step_signal(step_signal)
        self.clean_source = step_corrected_spectrum

    def get_sp_weights(self, beam):
        signal = np.copy(self.masked_source[beam])
        nan_indexes = np.where(np.isnan(signal))[0]
        populations = self.starlight_base

        if len(nan_indexes) > 0:
            valid_indexes = np.where(~np.isnan(signal))[0]
            first_index = valid_indexes[0]
            last_index = valid_indexes[-1]
            signal = signal[first_index : last_index + 1]
            populations = self.starlight_base[first_index : last_index + 1, :]

        nan_indexes = np.where(np.isnan(signal))[0]
        if len(nan_indexes) > 0:
            valid_indexes = np.where(~np.isnan(signal))[0]
            interp_func = interp1d(valid_indexes, signal[valid_indexes], kind="linear")
            signal[nan_indexes] = interp_func(nan_indexes)

        return nnls(populations, signal)

    def make_sp_synthesis(self, beam):
        synthesis = self.get_sp_weights(beam)
        residual = synthesis[1]
        synthetic_spectrum = np.dot(synthesis[0], self.starlight_base.T)

        return synthetic_spectrum, residual

    def cost_function(self, params, verbose=False):
        print(f"Trying params: {params}")
        fft_filter, wavelet_filter = params
        self.calculate_and_subtract_step_signal(fft_filter, wavelet_filter)

        self.mask_signal(self.clean_source)

        residuals = []
        for beam in np.arange(len(self.clean_source)):
            synthetic_spectrum, residual = self.make_sp_synthesis(beam)
            residuals.append(residual)
        if verbose:
            print(f"Sum of residuals: {np.sum(residuals)}")
            print(f"Avg residual: {np.mean(residuals)}")
            print(f"Std residual: {np.std(residuals)}")
            print("-----------------")
        return np.sum(residuals)

    def plot_step_sp_fit(self, beam):
        sythetic_data = self.make_sp_synthesis(beam)[0]
        plt.figure(figsize=(50, 20))
        plt.plot(
            self.corrected_wl[: len(sythetic_data)],
            self.input_signal[beam][: len(sythetic_data)],
            label="Original signal",
        )
        plt.plot(
            self.corrected_wl[: len(sythetic_data)],
            self.clean_source[beam][: len(sythetic_data)],
            label="Corrected signal",
        )
        plt.plot(
            self.corrected_wl[: len(sythetic_data)],
            self.bad_signal[beam][: len(sythetic_data)],
            label="Step signal",
        )
        plt.plot(
            self.corrected_wl[: len(sythetic_data)],
            sythetic_data,
            label="Synthetic reference",
        )
        plt.legend()
        plt.show()

    def optimize_and_remove_step_signal(
        self,
        initial_fft_filter,
        initital_wavelet_filter,
        initial_step_size,
        constraints,
        verbose=False,
        plot=False,
        beam=None,
    ):
        initial_params = [initial_fft_filter, initital_wavelet_filter]

        minimization_result = minimize(
            self.cost_function,
            initial_params,
            args={verbose},
            method="COBYLA",
            options={"rhobeg": initial_step_size},
            constraints=constraints,
        )

        optimal_fft = minimization_result.x[0]
        optimal_wavelet = minimization_result.x[1]
        print(f"Optimal fft: {optimal_fft}")
        print(f"Optimal wavelet: {optimal_wavelet}")
        print(f"Sum of residuals: {minimization_result.fun}")

        self.calculate_and_subtract_step_signal(optimal_fft, optimal_wavelet)
        if plot == True:
            self.plot_step_sp_fit(beam)

        return self.bad_signal
