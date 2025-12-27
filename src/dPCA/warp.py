import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def timewarp_neural(switch_data: pd.DataFrame, fr: np.ndarray, window_left: int = 14):
    # first normalize switch times
    time_windows = np.vstack(
        (
            switch_data["start_idx"].values,
            switch_data["cross_idx"].values,
            switch_data["end_idx"].values,
        )
    ).T
    normalized_times = (
        time_windows
        - (switch_data["cross_idx"].values)[:, np.newaxis]
        + window_left
        + 1
    )

    # now get median event time
    median_event_times = np.median(normalized_times, axis=0).astype(int)

    # get original timestamps to warmp
    original_timestamps = np.arange(1, fr.shape[1] + 1)

    # now warp each neuron's fr
    for neuron in range(fr.shape[2]):
        n_fr = fr[:, :, neuron]
        warped_frs = []
        for switch_idx in range(fr.shape[0]):
            # if unable to warp, just use original array
            try:
                warped_fr, _ = warp_time_axis(
                    original_timestamps,
                    normalized_times[switch_idx, :],
                    median_event_times,
                    n_fr[switch_idx, :],
                    warpevent=2,
                )
            except Exception as e:
                print(e)
                warped_fr = n_fr[switch_idx]
            warped_frs.append(warped_fr)
        warped_frs = np.array(warped_frs)
        fr[:, :, neuron] = warped_frs

    return fr, median_event_times


def warp_time_axis(
    original_times, event_times, median_event_times, firing_rates, warpevent=2
) -> np.ndarray:
    """
    Warp the time axis based on event times and median event times.

    Parameters:
        original_times (np.ndarray): Original time points.
        event_times (np.ndarray): Event times for the trial.
        median_event_times (np.ndarray): Median event times across trials.
        firing_rates (np.ndarray): Firing rates corresponding to the original time points.
        warpevent (int): Specifies the warping type (1 or 2).

    Returns:
        tuple: Warped firing rates and warped times.
    """
    warped_times = np.copy(original_times)  # Start with the original times
    warped_firing_rates = np.zeros_like(firing_rates)

    if warpevent == 1:
        # Warping based on the first event

        # Calculate scaling factors for each segment
        scale_factor1 = (median_event_times[0] - original_times[0]) / (
            event_times[0] - original_times[0]
        )
        scale_factor2 = (median_event_times[1] - median_event_times[0]) / (
            event_times[1] - event_times[0]
        )

        # Segment 1: Start to Event 1
        seg1 = original_times[original_times <= event_times[0]]
        warped_times[: len(seg1)] = (
            original_times[0] + (seg1 - original_times[0]) * scale_factor1
        )

        # Segment 2: Event 1 to Event 2
        seg2 = original_times[
            (original_times > event_times[0]) & (original_times <= event_times[1])
        ]
        seg2_warped_start = warped_times[len(seg1) - 1]
        warped_times[len(seg1) : len(seg1) + len(seg2)] = (
            seg2_warped_start + (seg2 - seg2[0]) * scale_factor2
        )

        # Segment 3: Event 2 to End
        seg3 = original_times[original_times > event_times[1]]
        seg3_warped_start = warped_times[len(seg1) + len(seg2) - 1]
        scale_factor3 = (original_times[-1] - seg3_warped_start) / (seg3[-1] - seg3[0])
        warped_times[len(seg1) + len(seg2) :] = (
            seg3_warped_start + (seg3 - seg3[0]) * scale_factor3
        )

    elif warpevent == 2:
        # Warping scaled to the 2nd event
        et = np.copy(event_times)
        if et[2] < 0:
            et[2] = median_event_times[2]  # Impute invalid event time

        # Time before the 2nd event
        scale_factor_before = (median_event_times[0] - original_times[0]) / (
            et[0] - original_times[0]
        )
        segment_before = np.arange(original_times[0], et[0] + 1).astype(int)

        warped_times[segment_before] = (
            original_times[0]
            + (segment_before - original_times[0]) * scale_factor_before
        )

        # Time after the 2nd event
        scale_factor_after = (original_times[-1] - median_event_times[2]) / (
            original_times[-1] - et[2]
        )
        segment_after = original_times[original_times >= et[2]]
        warped_times[-len(segment_after) :] = (
            et[2] + (segment_after - et[2]) * scale_factor_after
        )

        # Interpolate firing rates to match warped times
    if firing_rates.ndim == 1:
        interpolator = interp1d(
            original_times,
            firing_rates,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

        # Interpolate onto warped_times
        warped_firing_rates = interpolator(warped_times)

    else:
        for i in range(firing_rates.shape[0]):
            interpolator = interp1d(
                original_times,
                firing_rates[i],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )

            # Interpolate onto warped_times
            warped_firing_rates[i] = interpolator(warped_times)
            # warped_firing_rates[i] = np.interp(warped_times, original_times, firing_rates[i])

    return warped_firing_rates, warped_times
