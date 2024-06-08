import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from vorbin.voronoi_2d_binning import voronoi_2d_binning


class VorbinUtils:
    def __init__(self, filename, ini_w, fin_w):
        with fits.open(filename) as hdul:
            self.data = hdul["SCI"].data
            self.var = hdul["VAR"].data
            self.dq = hdul["DQ"].data

        fits_header = fits.getheader(filename, ext=("sci", 1))
        wcs = WCS(fits_header, naxis=[3])
        wavelength = wcs.wcs_pix2world(np.arange(fits_header["NAXIS3"]), 0)[0]

        self.ini_index = np.where(np.round(wavelength) == ini_w)[0][0]
        self.fin_index = np.where(np.round(wavelength) == fin_w)[0][0]

    def voronoi_bin(self, target_sn):
        sample = self.data[self.ini_index : self.fin_index, :, :]
        sample_var = self.var[self.ini_index : self.fin_index, :, :]
        sample_noise = np.sqrt(sample_var)

        signal = np.nanmean(sample, axis=0)
        noise = np.nanmean(sample_noise, axis=0) + 1e-10
        x, y = np.indices(signal.shape)
        x = x.ravel()
        y = y.ravel()
        bin_num, x_bin, y_bin, x_bar, y_bar, sn, n_pixels, scale = voronoi_2d_binning(
            x, y, signal.ravel(), noise.ravel(), target_sn, plot=0, quiet=1
        )

        self.x_bar = x_bar
        self.y_bar = y_bar

        threshold = 0.3
        self.bin_ids = list(set(bin_num))
        binned_data = {}
        for bin_id in self.bin_ids:
            binned_data[bin_id] = {}
            bin_mask = bin_num == bin_id
            binx = x[bin_mask]
            biny = y[bin_mask]
            bin_mask = np.zeros(self.data[0, :, :].shape, dtype=bool)

            signal_sum = np.zeros(self.data[:, 0, 0].shape)
            var_sum = np.zeros(self.var[:, 0, 0].shape)

            for pixx, pixy in zip(binx, biny):
                signal_sum += self.data[:, pixx, pixy]
                var_sum += self.var[:, pixx, pixy]
                bin_mask[pixx, pixy] = True

            binned_data[bin_id]["quality_map"] = np.zeros(self.data.shape[0])
            for wavl in np.arange(self.data.shape[0]):
                wavl_dq_plane = self.dq[wavl, :, :]
                bad_data_ratio = np.round(
                    np.sum(wavl_dq_plane[bin_mask]) / np.sum(bin_num == bin_id), 2
                )
                binned_data[bin_id]["quality_map"][wavl] = np.where(
                    bad_data_ratio >= threshold, 2, 0
                ).astype(int)
            bin_signal = signal_sum / len(binx)
            bin_var = var_sum / len(binx)
            binned_data[bin_id]["signal"] = bin_signal
            binned_data[bin_id]["var"] = bin_var

        self.binned_data = binned_data

    def calculate_true_snr(self, plot=False):
        true_snr = {}
        for bin_id in self.bin_ids:
            signal = np.mean(
                self.binned_data[bin_id]["signal"][self.ini_index : self.fin_index]
            )
            noise = np.mean(
                np.sqrt(
                    self.binned_data[bin_id]["var"][self.ini_index : self.fin_index]
                )
            )
            true_snr[bin_id] = signal / noise

        all_bins = list(true_snr.keys())
        x_all = np.array(self.x_bar)[all_bins]
        y_all = np.array(self.y_bar)[all_bins]
        snr_values = np.array(list(true_snr.values()))
        good_snr_indices = [index for index, snr in enumerate(snr_values) if snr > 5]
        print(f"Good bins: {len(good_snr_indices)}")
        print(f"Share of good bins: {np.round(len(good_snr_indices)/len(all_bins), 3)}")
        bad_snr_indices = [index for index, snr in enumerate(snr_values) if snr < 5]
        really_bad_snr_indices = [
            index for index, snr in enumerate(snr_values) if snr < 4
        ]

        if plot:
            # plt.figure(figsize=(15, 10))
            plt.scatter(
                x_all[bad_snr_indices],
                y_all[bad_snr_indices],
                c="black",
                marker="x",
                s=90,
                label="SNR < 5",
            )
            plt.scatter(
                x_all[really_bad_snr_indices],
                y_all[really_bad_snr_indices],
                c="red",
                marker="x",
                s=90,
                label="SNR < 4",
            )

            cmap = plt.cm.jet
            sc = plt.scatter(
                x_all,
                y_all,
                c=snr_values,
                cmap=cmap,
                vmin=min(snr_values),
                vmax=max(snr_values),
            )
            plt.colorbar(sc, ax=plt.gca(), label="SNR")
            plt.legend()
            plt.tight_layout()
            plt.gca().set_aspect("equal")
            plt.title("SNR heatmap - GSN 069")
            plt.xlabel("X coordinate")
            plt.ylabel("Y coordinate")
            plt.show()
