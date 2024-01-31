import inspect
import itertools
import multiprocessing
import os
import traceback
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class mcUNetPredictor(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        nnUNetPredictor.__init__(tile_step_size=tile_step_size, use_gaussian=use_gaussian, use_mirroring=use_mirroring,
                       perform_everything_on_device=perform_everything_on_device, device=device, verbose=verbose,
                       verbose_preprocessing=verbose_preprocessing, allow_tqdm=allow_tqdm)
        # The format of dataset_label_dict is {'dataset_name': [label1, label2, ...], ...}
        assert self.plans_manager.plan['dataset_label_dict'] is not None, "dataset_label_dict must be not None"
        self.dataset_label_dict = self.plans_manager.plan['dataset_label_dict']
        labels_num_per_dataset = [len(label_list) for label_list in self.dataset_label_dict.values()]
        labels_start_ind_per_dataset = [sum(labels_num_per_dataset[:i]) for i in range(len(labels_num_per_dataset))]
        labels_end_ind_per_dataset = [sum(labels_num_per_dataset[:i+1]) for i in range(len(labels_num_per_dataset))]

        self.dataset_label_start_end_dict = zip(self.dataset_label_dict.keys(), [[i,j] for i,j in zip(labels_start_ind_per_dataset, labels_end_ind_per_dataset)])
    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    print('sleeping')
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                # ofline must be not None if prediction is not to be returned
                assert ofile is not None or prediction is not None, "either ofile or prediction must be not None"

                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, configuration_manager, plans_manager,
                    #                               dataset_json, ofile, save_probabilities)
                    print('sending off prediction to background worker for resampling and export')

                    dataset_name_list = self.dataset_label_dict.keys()
                    dataset_start_ind = self.dataset_label_start_end_dict[dataset_name][0]
                    dataset_end_ind = self.dataset_label_start_end_dict[dataset_name][1]

                    for dataset_name,label_list in dataset_name_list.items():
                        cur_dataset_ofile = ofile.replace('.nii.gz', f'_{dataset_name}.nii.gz')
                        cur_prediction = prediction[dataset_start_ind:dataset_end_ind]
                        export_pool.starmap_async(
                                export_prediction_from_logits,
                                ((cur_prediction, properties, self.configuration_manager, self.plans_manager,
                                  self.dataset_json, cur_dataset_ofile, save_probabilities),)
                            )
                print(f'done with {os.path.basename(ofile)}')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

