# fiber-tracking
Individual fiber separation from tomography images (see https://doi.org/10.1016/j.commatsci.2024.113372)

Reguired libraries
* Skan (https://skeleton-analysis.org)
* SciPy
* Scikit-Image
* tqdm
* Pandas
* Matplotlib
* Mayavi (for visualization)


The repository includes a small cropped sample preprocessed binary image for testing. To run the tracking:

    python fiber_tracking.py example_data/PLA15_cropped.tif PLA15_cropped.zip

The tracking results will be saved as a zip file. To analyze the fibers, type

    python fiber_tracking.py example_data/PLA15_cropped.tif PLA15_cropped.zip

The script will create a file `PLA15_cropped_fiberdata.zip`. Some statistics can be extracted by

    python tables_and_plots.py PLA15_cropped_fiberdata.zip 

See the file `tables_and_plots.py` for various plotting functions.
