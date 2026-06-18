## Calculate echo base and top height for each frame of SEA-POL volume data 

import xarray as xr
import numpy as np
import pandas as pd

from scipy.interpolate import interp2d, RectBivariateSpline
from datetime import datetime, timedelta
import cftime
import json
import glob
import os

# read in data
seapol = xr.open_dataset('/huracan/tank4/cornell/ORCESTRA/sea-pol/qc_data/level4v1.2/PICCOLO_level4_volume_3D.nc')

# set to nan outside of radius 120 km to only include data with the 3D volume
radius_outer = 120  # km
radius_inner = 50

# set reflectivity threshold for valid echoes
threshold = 10

# Make regular 10-minute time series
start_time = np.datetime64('2024-08-16T08:00:00')
#start_time = np.datetime64('2024-09-13T00:00:00') #East/west division
end_time = np.datetime64('2024-09-23T16:50:00')
time10m = pd.date_range(start_time, end_time, freq='10 min')
time10m = pd.to_datetime(time10m)

# Loop over times
#for i in range(0, np.size(seapol.time[:-1])): #loop over all times
for i in range(0,len(time10m)):  # loop over all times in the 10-minute series (VOL1 only)

    #check if there is data for this time
    #if seapol.time[i].values not in seapol.time.values:
    if time10m[i] not in seapol.time.values: #VOL1 only
        #print(f"No data for time {seapol.time[i].values}")
        print(f"No data for time {time10m[i]}") #VOL1 only
        elevated_fraction1 = np.nan
        elevated_fraction2 = np.nan
        elevated_fraction3 = np.nan
        elevated_fraction4 = np.nan
    else:
        #print('Time:', seapol.time[indexAP + i].values)
        #map_dbz = seapol.DBZ[indexAP+i,:,:,:]
         
        print('Time:', seapol.time.sel(time=time10m[i]).values) #VOL1 only
        map_dbz = seapol.DBZ.sel(time=time10m[i]) #VOL1 only

        # set to nan outside of radius_outer to only include data with the 3D volume
        distances = np.sqrt((map_dbz.latitude - map_dbz.latitude[120, 120])**2 + (map_dbz.longitude - map_dbz.longitude[120, 120])**2) * 111.32  # Approximate conversion from degrees to km
        map_dbz = map_dbz.where(distances<=radius_outer,np.nan)  # Set values outside the radius to NaN

        # also mask within radius_inner, to limit to radial range where we have "full" 3D coverage
        #map_dbz = map_dbz.where(distances >= radius_inner, np.nan) ##SKIP this now because we are checking for confident tops

        # mask for valid (non-NaN) data
        valid_data = ~np.isnan(map_dbz.values)

        # mask for reflectivity above threshold
        above_thresh = valid_data & (map_dbz.values >= threshold)

        # Find the lowest index (height) where above_thresh is True for each (y, x)
        min_indices = np.where(above_thresh.any(axis=0), np.argmax(above_thresh, axis=0), -1)
        
        # Find the highest index (height) where above_thresh is True for each (y, x)
        max_indices = np.where(above_thresh.any(axis=0), above_thresh.shape[0] - 1 - np.argmax(above_thresh[::-1], axis=0), -1)

        # Initialize output: NaN where no valid data, -5 where threshold not met
        echo_base_height = np.full(map_dbz.shape[1:], np.nan)
        echo_top_height = np.full(map_dbz.shape[1:], np.nan)
        has_valid = valid_data.any(axis=0)
        echo_base_height[has_valid] = -5
        echo_top_height[has_valid] = -5
        
        # Set echo top height where threshold is met, checking there are possible echoes above that height (dBZ < threshold or -9999)
        valid_top = max_indices != -1 #this is a mask for where the threshold is met

        #also check that the maximum index is not at the top of the profile (nan above)
        above_max_indices = np.copy(max_indices)
        above_max_indices[valid_top] = above_max_indices[valid_top] + 1 # this is the index just above the maximum index where the threshold is met
        dBZabove = np.full(map_dbz.shape[1:], np.nan)

        #grab dbz value at the index above the maximum index where the threshold is met (if valid)
        valid_above = valid_top & (above_max_indices >= 0) & (above_max_indices < map_dbz.shape[0])
        yy, xx = np.where(valid_above)
        zz = above_max_indices[yy, xx].astype(int)
        dBZabove[yy, xx] = map_dbz.values[zz, yy, xx]

        #if the dbz value above the max index is not valid (nan), set the max index and min index to -1
        #basically also limiting echo base height to where there is a confident echo top height
        confident_max_indices = np.copy(max_indices)
        confident_max_indices = np.where(~np.isnan(dBZabove), confident_max_indices, -1)
        confident_min_indices = np.copy(min_indices)
        confident_min_indices = np.where(~np.isnan(dBZabove), confident_min_indices, -1)
        confident_valid_top = confident_max_indices != -1 #this is a mask for where the threshold is met and there is valid data above the max index
        confident_valid_base = confident_min_indices != -1 

        #grab the values of the height at the indices of the confident valid tops
        echo_top_height[confident_valid_top] = map_dbz.Z.values[confident_max_indices[confident_valid_top].astype(int)]
        echo_base_height[confident_valid_base] = map_dbz.Z.values[confident_min_indices[confident_valid_base].astype(int)]
        #echo_top_height[valid_top] = map_dbz.Z.values[max_indices[valid_top].astype(int)] #old way with no check if at top of scan

        # Reshape into a 1D array
        echo_base_height_flat = echo_base_height[confident_valid_base].flatten()
        echo_top_height_flat = echo_top_height[confident_valid_top].flatten()
        
        # Calculate fraction of echoes in this scene that are elevated
        elevated_fraction1 = len(np.where(echo_base_height_flat>1000)[0]) / len(echo_base_height_flat) if len(echo_base_height_flat) > 0 else np.nan
        elevated_fraction2 = len(np.where(echo_base_height_flat>2000)[0]) / len(echo_base_height_flat) if len(echo_base_height_flat) > 0 else np.nan
        elevated_fraction3 = len(np.where(echo_base_height_flat>3000)[0]) / len(echo_base_height_flat) if len(echo_base_height_flat) > 0 else np.nan
        elevated_fraction4 = len(np.where(echo_base_height_flat>4000)[0]) / len(echo_base_height_flat) if len(echo_base_height_flat) > 0 else np.nan

        # Concatenate the echo base heights into a single array
        if i == 0 or 'all_echo_base_heights' not in locals():
            all_echo_base_heights = echo_base_height_flat
            all_echo_top_heights = echo_top_height_flat
        else:
            all_echo_base_heights = np.concatenate((all_echo_base_heights, echo_base_height_flat))
            all_echo_top_heights = np.concatenate((all_echo_top_heights, echo_top_height_flat))
    
    #outside the else statement, so that elevated fraction is still recorded even if no data for that time
    if i == 0:
        all_elevated_fractions1 = np.array([elevated_fraction1])
        all_elevated_fractions2 = np.array([elevated_fraction2])
        all_elevated_fractions3 = np.array([elevated_fraction3])
        all_elevated_fractions4 = np.array([elevated_fraction4])
    else:
        all_elevated_fractions1 = np.concatenate((all_elevated_fractions1, np.array([elevated_fraction1])))
        all_elevated_fractions2 = np.concatenate((all_elevated_fractions2, np.array([elevated_fraction2])))
        all_elevated_fractions3 = np.concatenate((all_elevated_fractions3, np.array([elevated_fraction3])))
        all_elevated_fractions4 = np.concatenate((all_elevated_fractions4, np.array([elevated_fraction4])))


# Save the echo base heights to netcdf
ds = xr.Dataset(data_vars={'echo_base_height': (['data points'], all_echo_base_heights),
                'echo_top_height': (['data points'], all_echo_top_heights),
                'elevated_echo_fraction1': (['time'], all_elevated_fractions1),
                'elevated_echo_fraction2': (['time'], all_elevated_fractions2),
                'elevated_echo_fraction3': (['time'], all_elevated_fractions3),
                'elevated_echo_fraction4': (['time'], all_elevated_fractions4)},
                coords={'data points': np.arange(len(all_echo_base_heights)),
                        'time': time10m})
ds.to_netcdf('../../data/SEA-POLv1.2_echo_base_top_height_vol1_10dbz_confident.nc')