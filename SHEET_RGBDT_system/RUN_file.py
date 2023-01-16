import time



### run PROCESS scripts
start = time.time()
# RGBD-T alignment
exec(open('ALIGNMENT.py').read())
# temperature and position extraction of each detected object and output dataframe creation
exec(open('SHEET_Temp_Position_extractor_from_Yolo.py').read())
end = time.time()

# visualisation
exec(open('Visualise_orchard.py').read())
# modify the dataset to change XYZ axis
exec(open('reordering_output_CSV_for_DELIVERABLE.py').read())
print(f'\nminutes {(end - start)/60}')

