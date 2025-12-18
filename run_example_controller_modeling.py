from src.controller_modeling.pipeline import BehavPipeline
from src.controller_modeling.config import BehavConfig
from src.patient_data.session import PatientData
from src.patient_data.config import Config
import os

# create default configs
data_config = Config()
behav_config = BehavConfig()

# get path for our data
data_path = 'example_data/YFD'

# create patient data object and load data
patient_data = PatientData('YFD', data_path, data_config)
patient_data.load()
patient_data.compute_design_matrix()

controller_types = ['p', 'pv', 'pf', 'pi', 'pvi', 'pif', 'pvf', 'pvif']

for controller in controller_types:
    behav_config.model = controller
    # create behavior modeling pipeline obj
    behav_pipeline = BehavPipeline(patient_data, behav_config)
    
    # create modeling output path and run pipeline
    output_path = f'{data_path}/{controller}'
    try:
        os.mkdir(output_path)
    except Exception as e:
        print(e)
    behav_pipeline.run_model_pipeline(output_path)