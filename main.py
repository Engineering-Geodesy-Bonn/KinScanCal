from src.core.KinematicCalibration import KinematicCalibration
from src.pointcloud.dem import plot_dem_and_kinematic_calibration

from colorama import init, Fore, Style
import click
import shutil
import os
import glob

@click.command()
@click.option("--parent_dir", "-pa", default="/mnt/syn180/241111_FieldPheno4D_multi_crop_multi_modal/01_cropplotdata/New_structure", type=str, help="Path to the dataset directory")
@click.option("--output_dir", "-pb", default="output/", type=str, help="Path to the output data directory")
@click.option("--calibration_dir", "-pc", default="input/calibration/", type=str, help="Path to the static calibration of the laser scanners")
@click.option("--configfile", "-pd", default="config/kin_calibration_config.json", type=str, help="Config file of the kinematic calibration")
@click.option("--plot_id", "-pe", default="P144", type=str, help="Plot id to process")
@click.option("--date", "-pf", default="230516", type=str, help="Plot id to process")

def main(parent_dir,
         output_dir,
         calibration_dir,
         configfile,
         plot_id,
         date):
    
    # Directory of the current dataset
    dataset_dir = os.path.join(parent_dir, plot_id, date)
    
    #####################################################################################
    # 2) Initialization and data preparation

    # Initialize 
    kin_cal = KinematicCalibration( parent_dir,
                                    output_dir,
                                    calibration_dir,
                                    configfile )
    
    # Copy data from dataset path to local repo
    kin_cal.copy_data( plot_id, date )
    
    # Print dataset info
    kin_cal.print_info()

    # Load kinematic calibration config file
    kin_cal.loadconfig()

    # Load static calibration from path
    kin_cal.loadcalibration()

    # Load data from path
    kin_cal.loaddata()
    
    # Create initial point cloud with static calibration parameter (optional)
    #pcl, pcr = kin_cal.create_pointcloud( calibration = "static" )
    #pc_s = pcl.concatenate(pcr)

    #pc_s.write_to_file( path = kin_cal.output_dir, filename = "pc_static_calibration", offset = kin_cal.config.txyz )

    #####################################################################################
    # 2) Kinematic calibration

    # Run alignment
    kin_cal.run()

    # Compute kinematic calibration parameter
    kin_cal.compute_kinematic_calibration_parameter()

    #####################################################################################
    # 3) Create final point clouds with the time-dependent calibration parameter

    pcl, pcr = kin_cal.create_pointcloud( calibration = "kinematic" )

    # Merge point clouds to one
    pc = pcl.concatenate(pcr)

    # Write point clouds to file
    pc.write_to_file( path = kin_cal.output_dir, filename = "pc_kinematic_calibration", offset = kin_cal.config.txyz )

    #####################################################################################
    # 4) Copy files back to dataset folder structure (optional)
    
    # 4.1 Calibration files
    cal_files = glob.glob(os.path.join("output/", "*.txt"))

    # Copy each file
    for file_path in cal_files:
        filename = os.path.basename(file_path)
        destination_path = os.path.join(os.path.join(dataset_dir,"03_calibration"), filename)

        try:
            shutil.copy(file_path, destination_path)
        except Exception as e:
            continue

    # Create a combined DEM and calibration overview plot in the dataset folder
    plot_dem_and_kinematic_calibration(dataset_dir)

    # 4.2 Point clouds
    pc_files = glob.glob(os.path.join("output/", "*.las"))

    # Copy each file
    for file_path in pc_files:
        filename = os.path.basename(file_path)
        destination_path = os.path.join(os.path.join(dataset_dir,"04_pointcloud"), filename)
        try:
            shutil.copy(file_path, destination_path)
        except Exception as e:
            continue

if __name__ == "__main__":
    main()