# Read functions for various ORCESTRA datasets.
# 
# James Ruppert
# 18 Sept 2024

import numpy as np
import subprocess
import xarray as xr
import pandas as pd

# data_main = "/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/piccolo-data/data/"
data_main = "./data/"
orcestra_ipns_root = "ipns://latest.orcestra-campaign.org"

#############################################
### General read function for IPNS
#############################################

def get_ipfs_data(path, var):
    ds = xr.open_dataset(f"{orcestra_ipns_root}/products/{path}", engine="zarr")
    data = ds[var].data
    try:
        times = ds[var].time.data
    except:
        times = ds[var].TIME.data
    ds.close()
    return np.array(data), times



#############################################
### Mask sounding data
#############################################

# Mask soundings that don't reach (default) 100 hPa
# Not using actual masking, just setting to NaN
def mask_soundings(soundings, p_threshold=100): # p_threshold should be in hPa
    import copy
    # First save minimum pressure
    nt = soundings['p'].shape[0]
    min_pres = np.full(nt, np.nan)
    soundings_masked = copy.deepcopy(soundings)
    for isnd in range(nt):
        min_pres[isnd] = np.nanmin(soundings['p'][isnd,:]*1e-2) # Pa --> hPa
        if min_pres[isnd] > p_threshold:
            for key in soundings.keys():
                try:
                    soundings_masked[key][isnd,:] = np.nan
                except:
                    continue
    # Mask out soundings that don't reach 100 hPa
    # min_pres_2d = np.repeat(min_pres[:,np.newaxis], soundings['p'].shape[1], axis=1)
    # idx_masked1d = (min_pres > p_threshold).nonzero()
    # idx_masked2d = (min_pres_2d > p_threshold).nonzero()
    # soundings_masked = copy.deepcopy(soundings)
    # for key in soundings_masked.keys():
    #     if key == 'hght': continue
    #     elif soundings_masked[key].ndim == 2:
    #         # soundings[key] = np.ma.masked_where(idx_masked, soundings[key], copy=False)
    #         soundings_masked[key][idx_masked2d] = np.nan
    #     elif soundings_masked[key].ndim == 1:
    #         # soundings[key] = np.ma.masked_where(idx_masked[:,0], soundings[key], copy=False)
    #         soundings_masked[key][idx_masked1d] = np.nan
    #     else:
    #         raise ValueError(f"Unexpected number of dimensions for {key}")
    return soundings_masked, min_pres



#############################################
### Meteor sounding data
#############################################

#### File and time list

# def get_sounding_filelist(search_string):

#     main = data_main+'soundings/level2/'

#     process = subprocess.Popen(['ls --color=none '+main+'*'+search_string+'*nc'],shell=True,
#         stdout=subprocess.PIPE,universal_newlines=True)
#     snd_files = process.stdout.readlines()
#     nsnd=len(snd_files)
#     times=[]
#     for ifile in range(nsnd):
#         snd_files[ifile] = snd_files[ifile].strip()
#         time_str = snd_files[ifile].split('/')[-1].split('.')[0]
#         # print(time_str)
#         mm = time_str[-12:-10]
#         dd = time_str[-10:-8]
#         hh = time_str[-7:-5]
#         nn = time_str[-5:-3]
#         sounding_time = np.datetime64('2024-'+mm+'-'+dd+'T'+hh+':'+nn)
#         if search_string == 'ascen':
#             # Add 1:10 to all times
#             sounding_time += np.timedelta64(70, 'm')
#         times.append(sounding_time)

#     return snd_files, np.array(times)

#### Add dummy time steps for big jumps in time

def fix_time_3hrly(times_in, files_in):

    time_0 = np.datetime64('2024-08-14T21:00')
    time_1 = np.datetime64('2024-10-05T00:00')
    delta_3h = np.timedelta64(180, 'm')
    times_new = np.arange(time_0, time_1, delta_3h, dtype='datetime64[m]')
    ntime = len(times_new)
    # Assumes the sounding time is within one hour of 3h time stamp
    delta_check = np.timedelta64(60, 'm')

    files_new = []
    ntime_new=0
    for itime in range(ntime):
        tdiff = np.abs(times_in - times_new[itime])
        tdiff_min = tdiff.min()
        if tdiff_min <= delta_check:
            file_itime = np.where(tdiff == tdiff.min())[0][0]
            files_new.append(files_in[file_itime])
            ntime_new=itime+1
        else:
            files_new.append('null')
    times_new=times_new[:ntime_new]
    files_new=files_new[:ntime_new]

    return files_new, times_new

#### Main variable read loop

# def read_bowtie_soundings(search_string = 'ascen'):
def read_soundings(platform='RV_Meteor', ascent=0):

    path = 'Radiosondes/Level_2/RAPSODI_RS_ORCESTRA_level2.zarr'
    sndfile = xr.open_dataset(f"{orcestra_ipns_root}/products/{path}", engine="zarr")
    # Print unique platforms
    # print(np.unique(platform))
    # hght = np.squeeze(sndfile['alt'].data) # m
    hght = np.squeeze(sndfile['height'].data) # m
    platform = sndfile.variables['platform'].data
    ascent_flag = sndfile.variables['ascent_flag'].data # 0, 1 --> ascent, descent
    isondes = (platform == 'RV_Meteor') & (ascent_flag == ascent)
    times = sndfile['launch_time'][isondes].data
    nt = len(times)

    p    = np.ma.masked_invalid(np.squeeze(sndfile['p'][isondes].data))    # Pa
    tmpk = np.ma.masked_invalid(np.squeeze(sndfile['ta'][isondes].data))   # K
    rh   = np.ma.masked_invalid(np.squeeze(sndfile['rh'][isondes].data))*1e2 # 0-1 --> %
    mr   = np.ma.masked_invalid(np.squeeze(sndfile['mr'][isondes].data))   # kg/kg
    # q[ifile,:]    = np.ma.masked_invalid(np.squeeze(sndfile['q'].data))   # kg/kg
    wdir = np.ma.masked_invalid(np.squeeze(sndfile['wdir'][isondes].data)) # deg
    u    = np.ma.masked_invalid(np.squeeze(sndfile['u'][isondes].data))    # m/s
    v    = np.ma.masked_invalid(np.squeeze(sndfile['v'][isondes].data))    # m/s
    sndfile.close()
    hght_0c= np.full(nt, np.nan)
    for isnd in range(nt):
        try:
            hght_0c[isnd]= hght[ np.where(tmpk[isnd,:] <= 273.15)[0][0] ]
        except:
            continue
        hght_0c[isnd]= hght[ np.where(tmpk[isnd,:] <= 273.15)[0][0] ]
    inan = np.where(hght_0c > 8000)[0]
    hght_0c[inan] = np.nan
    lat= np.ma.masked_invalid(np.squeeze(sndfile['launch_lat'][isondes].data))   # deg N
    lon= np.ma.masked_invalid(np.squeeze(sndfile['launch_lon'][isondes].data))   # deg

    soundings = {
        'times': times,
        'lon':lon,
        'lat':lat,
        'hght':hght,
        'hght_0c':np.ma.masked_invalid(hght_0c),
        'p': p,
        'tmpk': tmpk,
        'rh': rh,
        'mr': mr,
        # 'q': q,
        'u': u,
        'v': v,
        'wdir': wdir,
    }

    return soundings



#############################################
### HALO dropsonde soundings
#############################################

def read_halo_soundings_lev3():
    path = "HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr"
    ds = xr.open_dataset(f"{orcestra_ipns_root}/products/{path}", engine="zarr")
    time = ds["launch_time"].data
    qual_flag = ds["sonde_qc"].data # 0, 1, 2 --> good, bad, ugly
    # snd_halo["qual_flag"] = qual_flag
    # Put variables into dictionary
    snd_halo = {}
    snd_halo["times"] = time
    snd_halo["hght"] = np.ma.masked_invalid(ds["altitude"].data) # m
    # snd_halo["iwv"]  = np.ma.masked_invalid(ds["iwv"].data) # kg/m^2
    # snd_halo["lat"]  = np.ma.masked_invalid(ds["aircraft_latitude"].data) # deg
    # snd_halo["lon"]  = np.ma.masked_invalid(ds["aircraft_longitude"].data) # deg
    snd_halo["lat"]  = np.ma.masked_invalid(ds["launch_lat"].data) # deg
    snd_halo["lon"]  = np.ma.masked_invalid(ds["launch_lon"].data) # deg
    snd_halo["p"]    = np.ma.masked_invalid(ds["p"].data) # Pa
    sh               = np.ma.masked_invalid(ds["q"].data) # specific humidity, kg/kg
    # Convert SH to MR for consistency with METEOR soundings
    # mr = sh / (1 - sh)
    snd_halo["mr"] = sh / (1 - sh) # mixing ratio, kg/kg
    snd_halo["tmpk"] = np.ma.masked_invalid(ds["ta"].data) # K
    snd_halo["u"]    = np.ma.masked_invalid(ds["u"].data) # m/s
    snd_halo["v"]    = np.ma.masked_invalid(ds["v"].data) # m/s
    ds.close()
    # Drop values depending on quality flag
    flag_value = 0
    indices1d = (qual_flag == flag_value).nonzero()
    indices1d_not = (qual_flag != flag_value).nonzero()
    time = np.array(time)[indices1d]
    for key in snd_halo.keys():
        if key == "hght":
            continue
        if snd_halo[key].ndim == 2:
            snd_halo[key] = np.delete(snd_halo[key], indices1d_not, axis=0)
        elif (snd_halo[key].ndim == 1) and (snd_halo[key].shape[0] == qual_flag.shape[0]):
            snd_halo[key] = snd_halo[key][indices1d]
        else:
            raise ValueError(f"Unexpected number of dimensions for {key}: shape is {snd_halo[key].shape}")
    return snd_halo

def read_halo_soundings_lev4():
    path = "HALO/dropsondes/Level_4/PERCUSION_Level_4.zarr"
    ds = xr.open_dataset(f"{orcestra_ipns_root}/products/{path}", engine="zarr")
    time = ds["circle_time"].data # m
    qual_flag = ds["sonde_qc"].data # 0, 1, 2 --> good, bad, ugly
    # snd_halo["qual_flag"] = qual_flag
    # Put variables into dictionary
    snd_halo = {}
    snd_halo["times"] = time
    snd_halo['sondes_per_circle'] = np.ma.masked_invalid(ds["sondes_per_circle"].data) # n
    snd_halo['circle_id'] = ds["circle_id"].data # #
    snd_halo["lat"]  = np.ma.masked_invalid(ds["circle_lat"].data) # deg
    snd_halo["lon"]  = np.ma.masked_invalid(ds["circle_lon"].data) # deg
    snd_halo['circle_altitude'] = np.ma.masked_invalid(ds["circle_altitude"].data) # m
    snd_halo['circle_radius'] = np.ma.masked_invalid(ds["circle_radius"].data) # #
    snd_halo['div'] = np.ma.masked_invalid(ds["div"].data) # /s
    snd_halo['omega'] = np.ma.masked_invalid(ds["omega"].data) # hPa/hr
    snd_halo['w'] = np.ma.masked_invalid(ds["wvel"].data) # m/s
    snd_halo['vor'] = np.ma.masked_invalid(ds["vor"].data) # /s
    snd_halo["hght"] = np.ma.masked_invalid(ds["altitude"].data) # m
    # snd_halo["iwv"]  = np.ma.masked_invalid(ds["iwv"].data) # kg/m^2
    snd_halo["p"]    = np.ma.masked_invalid(ds["p_mean"].data) # Pa
    sh = np.ma.masked_invalid(ds["q_mean"].data) # specific humidity, kg/kg
    # Convert SH to MR for consistency with METEOR soundings
    # mr = sh / (1 - sh)
    snd_halo["mr"] = sh / (1 - sh) # mixing ratio, kg/kg
    snd_halo["tmpk"] = np.ma.masked_invalid(ds["ta_mean"].data) # K
    snd_halo["u"]    = np.ma.masked_invalid(ds["u_mean"].data) # m/s
    snd_halo["v"]    = np.ma.masked_invalid(ds["v_mean"].data) # m/s
    ds.close()
    # Drop values depending on quality flag
    # flag_value = 0
    # indices1d = (qual_flag == flag_value).nonzero()
    # indices1d_not = (qual_flag != flag_value).nonzero()
    # time = np.array(time)[indices1d]
    # for key in snd_halo.keys():
    #     if key == "hght":
    #         continue
    #     if snd_halo[key].ndim == 2:
    #         snd_halo[key] = np.delete(snd_halo[key], indices1d_not, axis=0)
    #     elif (snd_halo[key].ndim == 1) and (snd_halo[key].shape[0] == qual_flag.shape[0]):
    #         snd_halo[key] = snd_halo[key][indices1d]
    #     else:
    #         raise ValueError(f"Unexpected number of dimensions for {key}: shape is {snd_halo[key].shape}")
    return snd_halo



#############################################
### BCO soundings
#############################################

# def read_bco_soundings(descent = False):
#     path = "BCO/RS_BCO_level2.zarr"
#     ds = xr.open_dataset(f"{orcestra_ipns_root}/products/{path}", engine="zarr")
#     # time = ds['flight_time']
#     time = ds['launch_time']
#     ascent_flag = ds["ascent_flag"].data # 0, 1 --> ascent, descent
#     # Put variables into dictionary
#     snd_bco = {}
#     snd_bco["hght"] = np.ma.masked_invalid(ds["alt"].data) # m
#     # snd_bco["iwv"]  = np.ma.array(ds["iwv"].data) # kg/m^2
#     snd_bco["lat"]  = np.ma.masked_invalid(ds["lat"].data) # deg
#     snd_bco["lon"]  = np.ma.masked_invalid(ds["lon"].data) # deg
#     snd_bco["p"]    = np.ma.masked_invalid(ds["p"].data) # Pa
#     # snd_bco["q"]    = np.ma.masked_invalid(ds["q"].data) # specific humidity, kg/kg
#     snd_bco["mr"]   = np.ma.masked_invalid(ds["mr"].data) # mixing ratio, kg/kg
#     snd_bco["tmpk"] = np.ma.masked_invalid(ds["ta"].data) # K
#     snd_bco["u"]    = np.ma.masked_invalid(ds["u"].data) # m/s
#     snd_bco["v"]    = np.ma.masked_invalid(ds["v"].data) # m/s
#     ds.close()
#     # Drop values depending on ascent/descent flag
#     if descent:
#         flag_value = 1 # decending
#     else:
#         flag_value = 0 # ascending
#     indices1d = (ascent_flag == flag_value).nonzero()
#     indices1d_not = (ascent_flag != flag_value).nonzero()
#     time = np.array(time)[indices1d]
#     for key in snd_bco.keys():
#         if key == "hght":
#             continue
#         if snd_bco[key].ndim == 2:
#             snd_bco[key] = np.delete(snd_bco[key], indices1d_not, axis=0)
#         elif (snd_bco[key].ndim == 1) and (snd_bco[key].shape[0] == time.shape[0]):
#             snd_bco[key] = snd_bco[key][indices1d]
#         else:
#             raise ValueError(f"Unexpected number of dimensions for {key}")
#     return snd_bco, time



#############################################
### Radiometer data
#############################################

# Now downloading this data from IPFS

# def read_bowtie_radiometer():

#     # main = "/Volumes/wiss/M203/Radiometer_MWR-HatPro-Uni-Leipzig/Data/"
#     main = data_main+'radiometer/'

#     # process = subprocess.Popen(['ls --color=none '+main+'*/*singl*nc'],shell=True,
#     process = subprocess.Popen(['ls --color=none '+main+'*singl*nc'],shell=True,
#         stdout=subprocess.PIPE,universal_newlines=True)
#     rdm_files = process.stdout.readlines()
#     nfiles=len(rdm_files)
#     for ifile in range(nfiles):
#         rdm_files[ifile] = rdm_files[ifile].strip()
#         rdmfile = xr.open_dataset(rdm_files[ifile])
#         rdm_time = rdmfile['time'].data
#         cwv = rdmfile['iwv'].data
#         flag = rdmfile['iwv_quality_flag'].data
#         rdmfile.close()
#         rdm_time = np.array(rdm_time, dtype='datetime64[s]')
#         cwv = np.array(cwv)
#         cwv[np.where(flag != 0)] = np.nan
#         if ifile == 0:
#             times=rdm_time
#             cwv_rdm=cwv
#         else:
#             times=np.concatenate((times,rdm_time))
#             cwv_rdm=np.concatenate((cwv_rdm,cwv))

#     return cwv_rdm, times



#############################################
### Sun photometer data
#############################################

# Same data is now on IPFS, so take from there

# Downloading this data from https://aeronet.gsfc.nasa.gov/new_web/cruises_v3/Meteor_24_0.html

# def read_bowtie_sunphotometer():

#     # main_photometer = "/Volumes/wiss/M203/microtops/downloaded/Meteor_24_0/AOD/Meteor_24_0_all_points.lev15"
#     # main = data_main+"microtops/Meteor_24_0_old/AOD/Meteor_24_0_all_points.lev15"
#     main = data_main+"microtops/Meteor_24_0/AOD/Meteor_24_0_all_points.lev20"

#     photom = pd.read_csv(main, sep=',', on_bad_lines='skip', skiprows=4)

#     # Get Datetimes from time stamps
#     df_datetime = pd.DataFrame({'year': photom['Date(dd:mm:yyyy)'].str[-4:],
#                                 'month': photom['Date(dd:mm:yyyy)'].str[3:5],
#                                 'day': photom['Date(dd:mm:yyyy)'].str[0:2],
#                                 'hour': photom['Time(hh:mm:ss)'].str[0:2],
#                                 'minute': photom['Time(hh:mm:ss)'].str[3:5],
#                                 'second': photom['Time(hh:mm:ss)'].str[6:8]})

#     photom['Date(dd:mm:yyyy)'] = pd.to_datetime(df_datetime)

#     # Sort dataframe
#     photom = photom.sort_values('Date(dd:mm:yyyy)')

#     # Convert IWV column to float
#     photom['Water Vapor(cm)'] = pd.to_numeric(photom['Water Vapor(cm)'], errors='coerce')*10 # cm --> mm

#     # photom = pd.read_csv(main_photometer, sep=',', on_bad_lines='skip', skiprows=2)

#     # # Get Datetimes from time stamps
#     # df_datetime = pd.DataFrame({'year': photom['DATE'].str[-4:],
#     #                             'month': photom['DATE'].str[0:2],
#     #                             'day': photom['DATE'].str[3:5],
#     #                             'hour': photom['TIME'].str[0:2],
#     #                             'minute': photom['TIME'].str[3:5],
#     #                             'second': photom['TIME'].str[6:8]})
#     # photom['DATE'] = pd.to_datetime(df_datetime)
#     # # for icol in range(32):
#     # #     print(photom.iloc[0:3, icol])

#     # # Sort dataframe
#     # photom = photom.sort_values('DATE')

#     # # Convert IWV column to float
#     # photom['WATER'] = pd.to_numeric(photom['WATER'], errors='coerce')*10 # cm --> mm

#     return photom



#############################################
### SeaSnake data
#############################################

def read_bowtie_seasnake():

    # def read_seasnake_raw():
    # main = "/Volumes/wiss/M203/SeaSnake/seaSnakeData/"
    main = data_main+"SeaSnake/seaSnakeData/"

    process = subprocess.Popen(['ls --color=none '+main+'*/*dat'],shell=True,
        stdout=subprocess.PIPE,universal_newlines=True)
    dat_files = process.stdout.readlines()

    # Skip first day
    dat_files=dat_files[3:]
    nfile=len(dat_files)
    frames = []
    for ifile in range(nfile):
        idatfile = dat_files[ifile].strip()
        df = pd.read_csv(idatfile, header=None, sep=',', on_bad_lines='skip')
        frames.append(df)

    # Concatenate
    frames = pd.concat(frames)
    # Convert first row to datetime64
    frames[0] = pd.to_datetime(frames[0])

    # Convert columns to numeric
    ncolumns = frames.shape[1]
    for icol in range(1,ncolumns):
        frames[icol] = pd.to_numeric(frames[icol], errors='coerce')
    # Take care of some funky lines
    frames.loc[frames[2] > 200,2] = np.nan
    frames.loc[frames[4] > 200,4] = np.nan
    frames.loc[frames[4] < 10,4] = np.nan

    # Return concatenated dataframe sorted by time
    return frames.sort_values(0)



#############################################
### DSHIP data
#############################################

def read_bowtie_dship():

    file = data_main+'DSHIP/meteor_meteo_dship_20240923.nc'
    # dset=xr.open_dataset(file,engine='h5netcdf',chunks='auto')

    dset=xr.open_dataset(file)
    time = dset.time.data
    sst1 = dset.sst_port.data # Weatherstation
    sst2 = dset.sst_extern_port.data # Thermosalinigraph SBE38(DShip)
    sst3 = dset.sst_intern_port.data # Thermosalinigraph SBE38(DShip)
    wspd = dset.wspd.data # m/s
    shortwave = dset.swr.data
    dset.close()

    dship = {
        'time':time,
        'sst1':sst1,
        'sst2':sst2,
        'sst3':sst3,
        'wspd':wspd,
        'shortwave':shortwave,
        }

    return dship



#############################################
### Thermosalinograph SST data
#############################################

def read_thermosalin_sst():

    file = data_main+'DSHIP/M203_surf_oce.nc'

    dset=xr.open_dataset(file)
    time = dset.TIME.data
    sst = dset.TEMP.data # Thermosalinigraph 1 & 2
    dset.close()

    tsg_sst = {
        'time':time,
        'sst':sst,
        }

    return tsg_sst



#############################################
### ISAR SeaSkinTemp data
#############################################

def read_bowtie_ISAR_sst():

    main = data_main+"ISAR_seaskintemp/"

    process = subprocess.Popen(['ls --color=none '+main+'*ISAR*nc'],shell=True,
        stdout=subprocess.PIPE,universal_newlines=True)
    dat_files = process.stdout.readlines()
    nfile=len(dat_files)
    for ifile in range(nfile):
        dat_files[ifile] = dat_files[ifile].strip()
        sstfile = xr.open_dataset(dat_files[ifile])
        sst_time = sstfile['time'].data
        isst = sstfile['sea_surface_temperature'].data # K
        # flag = sstfile['iwv_quality_flag'].data
        sstfile.close()
        sst_time = np.array(sst_time, dtype='datetime64[s]')
        isst = np.array(isst)
        # cwv[np.where(flag != 0)] = np.nan
        if ifile == 0:
            times=sst_time
            sst=isst
        else:
            times=np.concatenate((times,sst_time))
            sst=np.concatenate((sst,isst))

    return sst, times



#############################################
### AEW tracks
#############################################

def read_aew_tracks():

    main = data_main+"AEW_tracks_BOWTIE_post_processed.nc"

    aewfile = xr.open_dataset(main)
    time = aewfile['time'].data
    # Dimensions are n-system (25) x ntime
    lon = aewfile['AEW_lon_smooth'].data
    # Dimensions are n-system (25) x ntime
    lat = aewfile['AEW_lat_smooth'].data
    aewfile.close()

    return lon, lat, time



#############################################
### IMERG precip
#############################################

def read_bowtie_imerg_precip():

    main = data_main+"meteor_IMERG_range1.2deg.nc"

    file = xr.open_dataset(main)
    time = file['time'].data
    # Dimensions are ntime
    imerg_mean = file['mean_precipitation'].data # mm/hr
    imerg_max = file['max_precipitation'].data # mm/hr
    file.close()

    return imerg_mean, imerg_max, time



#############################################
### Disdrometer precip
#############################################

# NOW GETTING THIS OFF OF IPFS

# def read_bowtie_disd_precip():

#     main = data_main+"disdrometer_RES_meteor2_joined.nc"

#     file = xr.open_dataset(main)
#     time = file['time'].data
#     # Dimensions are ntime
#     rainrate = file['rain_rate_qc'].data # mm/hr
#     file.close()

#     return rainrate, time



#############################################
### ERA5 IWV along ship track
#############################################

def read_era5_shiptrack_iwv():

    main = data_main+"ERA5_IWV_alongMETEORtrack.nc"

    file = xr.open_dataset(main)
    # time = file['time'].data
    time = file['time'].data
    time_sonde = file['time_sonde'].data
    # Dimensions are ntime
    iwv_alongtrack = file['PW_alongtrack_era5'].data # kg m^{-2}
    iwv_alongsonde = file['PW_alongsondes_era5'].data # kg m^{-2}
    file.close()

    return iwv_alongtrack, iwv_alongsonde, time, time_sonde
