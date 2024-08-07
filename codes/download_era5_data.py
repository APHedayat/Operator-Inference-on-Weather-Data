
# import modules
import apache_beam   # Needs to be imported separately to avoid TypingError
import xarray as xr
import logging
import os

# set up the log file
log_name = "./job_run.log"
if os.path.exists(f'{log_name}'):
    os.remove(f'{log_name}')
logging.basicConfig(filename=f'{log_name}', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.info("program started and logger was set up.")

# access data on cloud
logger.info("\naccessing data on cloud...")
data_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr/'
# data_path = "gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr/"
data = xr.open_zarr(data_path)
logger.info("done.\n")

# define variables needed
vars = ['2m_temperature', '10m_u_component_of_wind',
        '10m_v_component_of_wind', 'mean_sea_level_pressure']

# truncate the original data
data_vars = data[vars]

# save the zarr file to disk
logger.info("saving data to disk...")
data_vars.to_zarr('/scratch/engin_root/engin1/ahedayat/data/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr')
logger.info("done.\n")
