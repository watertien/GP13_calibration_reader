# Convert the original calibration format (in root)
# to python-friendly numpy.ndarray
import numpy as np
from scipy.fft import fft, fftfreq
import uproot as ur

calibration_dtype = [("du_id", int),
                     ("du_datetime", "datetime64[s]"),
                     ("trace", float, (4, 1024))]
# The frequency axis for FFT plots
trace_f = fftfreq(1024, d=2)[:512] * 1e3 # Unit: MHz

def nest_strip(f):
    """Strip the unnecessary nested struction of calibration data from ROOT
    
    Parameters:
    -----------
    f : dictionary
    The dictionary with the required infos of triggers.
      - du_id      : ID of the triggered DU
      - du_seconds : the timestamp of the trigger time
      - trace      : the four channels of ADC counts

    Returns:
    ----------
    trigger_arr : np.array
    All the trigger infos in a simpler structure.
    """
    n_trigger = len(f["du_id"])
    # Select all the 'good' triggers with the ADC count consistent with the unsigned int dtype
    mask_trigger_good = np.ones(n_trigger, dtype=bool)
    for i in range(4):
        trace_length = np.array([len(j[0]) for j in f[f"trace_{i}"]])
        # Add initial parameter for empty arrays
        trace_max = np.array([np.max(np.abs(j[0]), initial=0) for j in f[f"trace_{i}"]])
        mask_trigger_good = mask_trigger_good & (trace_length == 1024) & (trace_max <= 2**13)
    n_trigger_good = np.sum(mask_trigger_good)
    if n_trigger_good < 1:
      return np.array([], dtype=calibration_dtype)
    trigger_arr = np.zeros(n_trigger_good, dtype=calibration_dtype)
    # ID of detector units, e.g. 1080 stands for DU80
    du_id = f["du_id"].astype(int)[mask_trigger_good]
    # DU timestamps for each trigger
    du_time = f["du_seconds"][mask_trigger_good]
    # Get datetime from timestamps
    du_datetime = np.datetime64('1970-01-01T00:00:00') \
                + du_time.astype(int) * np.timedelta64(1, 's')
    trace_arr = np.zeros((n_trigger_good, 4, 1024))
    for i in range(4):
      # The 0-th element is the trace of the given DU.
      # TODO what about the mask to exclude the abnormal data?
      trace_arr[:,i,:] = [j[0] for j in f[f"trace_{i}"][mask_trigger_good]]
    # Assemle the output array
    trigger_arr["du_id"] = du_id
    trigger_arr["du_datetime"] = du_datetime
    trigger_arr["trace"] = trace_arr
    return trigger_arr
  

def root_reader(fname_list):
    """Read the root files given by the list

    Parameters:
    -----------
    fname_list: list
    the list of the root files

    Returns:
    -----------
    trigger_arr: numpy.ndarray
    the trigger infos for calibration mode
    """
    trigger_arr = np.array([], dtype=calibration_dtype)
    expression_list = ["du_id", "du_seconds", "trace_0", "trace_1", "trace_2", "trace_3"]
    for batch in ur.iterate([i + ":teventadc" for i in fname_list],
                            expressions=expression_list, library="np", allow_missing=True):
        trigger_arr = np.append(trigger_arr, nest_strip(batch))
    return trigger_arr