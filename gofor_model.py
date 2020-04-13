import collections
import logging
import os
import warnings
import numpy as np
from osgeo import gdal
import scipy
import hazelbean as hb
import hazelbean.pyramids
from hazelbean.ui import model, inputs
import pygeoprocessing
from collections import OrderedDict
from fiona import _shim, schema

import pandas as pd
import math

logging.basicConfig(level=logging.WARNING)
hb.ui.model.LOGGER.setLevel(logging.WARNING)
hb.ui.inputs.LOGGER.setLevel(logging.WARNING)

L = hb.get_logger('seals', logging_level='warning')
L.setLevel(logging.INFO)

logging.getLogger('Fiona').setLevel(logging.WARNING)
logging.getLogger('fiona.collection').setLevel(logging.WARNING)

np.seterr(divide='ignore', invalid='ignore')

# QUIRKY LINE: Because RST is called via a bat file it changes the CWD, but this means it misses the base_data dir. Thus, reset it with the location of THIS script.
file_project_dir = os.path.split(__file__)[0]

p = hb.ProjectFlow(project_dir=file_project_dir)

# questions_for_jorge
# 1 Is it safe to assume that this will always be for a specific site? With 1 shapefile?
# I will propose to Jorge the model flow with huge detail, then note that addons/modificaitons cant be allowed past a certain point.
# Interpreter can be found at C:\OneDrive\Repositories\rst\src\WinPython\python-3.6.5.amd64\python.exe src/rst_model.py
"""
TODO

RECIOMPILE and test executable, then email jorge with changelist.

TEST For other parameter, distance threshold, they played with it from 5km to other, found that it returns different values in output raster. on last step before output raster, normalize by highest value so that it is not affected by distance thershold. 

On outputs, need to implement the thresholds. Hard to just set a percentile, so jorge suggested that before normalizing the input, Can specify it out by quantiles, . thus make a new output in addition to the clipped normalized continuous, also have split to 10 quantiles, output int raster of decile.
  RESPONSE: Assumed u meant percentiles OF THE LAND THAT COULD BE RESTORED.
  
Make sure that intermediate files be replaced.
   COULD NOT REPLICATE

Spaslh screen and fix about.
OPTIONAL
'__suffix' added to end thus make it save the FILE with a SUFFIX but in the same workspace. 
    NO TIME


"""
### Science Equations
def standardized_ecological_uncertainty(percent_of_overall_forest_cover_within_distance_threshold):
    return (1 - ((1.37595 - 0.23498 * np.log(percent_of_overall_forest_cover_within_distance_threshold + 1) - 0.291489) / 1.084461)) * 100.0

### UI Helper functions
def launch_from_ui(p):
    from hazelbean.ui import model, inputs
    import gofor_ui

    clip_lulc_task = p.add_task(clip_lulc)
    resample_lulc_task = p.add_task(resample_lulc)
    reclassify_lulc_task = p.add_task(reclassify_lulc)
    calc_percent_of_overall_forest_cover_within_distance_threshold_task = p.add_task(calc_percent_of_overall_forest_cover_within_distance_threshold)
    calc_standardized_ecological_uncertainty_task = p.add_task(calc_standardized_ecological_uncertainty)

    ui = gofor_ui.GoforUI(p)
    ui.run()
    EXITCODE = inputs.QT_APP.exec_()  # Enter the Qt application event loop. Without this line the UI will launch and then close.

# UNNEEDED???
def launch_from_command_line(p):
    clip_lulc_task = p.add_task(clip_lulc)
    resample_lulc_task = p.add_task(resample_lulc)
    reclassify_lulc_task = p.add_task(reclassify_lulc)
    calc_percent_of_overall_forest_cover_within_distance_threshold_task = p.add_task(calc_percent_of_overall_forest_cover_within_distance_threshold)
    calc_standardized_ecological_uncertainty_task = p.add_task(calc_standardized_ecological_uncertainty)
    p.execute()

def circle_from_ogrid(convolution_array, convolution_edge_size):
    radius = (convolution_edge_size - 1) / 2
    a, b = radius, radius
    y, x = np.ogrid[-a: convolution_edge_size - a, -b:convolution_edge_size - b]
    mask = x * x + y * y <= radius ** 2

    convolution_array[mask] = 1
    return convolution_array

### TASKS
def clip_lulc():
    global p
    aoi_bb = hb.get_vector_info_hb(p.area_of_interest_path)['bounding_box']
    p.clipped_lulc_path = os.path.join(p.cur_dir, 'clipped_lulc.tif')

    hb.clip_raster_by_vector(p.input_lulc_path, p.clipped_lulc_path, p.area_of_interest_path)
    # hb.clip_raster_by_bb(p.input_lulc_path, aoi_bb, output_path)

def resample_lulc():
    global p
    # resample_method = 'near'
    # target_bb = None
    # base_sr_wkt = None
    # target_sr_wkt = None
    # gtiff_creation_options = hb.DEFAULT_GTIFF_CREATION_OPTIONS
    # n_threads = None
    # vector_mask_options = None
    # output_data_type = None
    # src_ndv = None
    # dst_ndv = None
    # calc_raster_stats = False
    # add_overviews = False
    p.resampled_lulc_path = os.path.join(p.cur_dir, 'resampled_lulf.tif')

    # Extract from the patch size the implied resolution
    p.resampling_threshold = 100.0 * float(p.minimum_patch_size) ** 0.5

    hb.warp_raster_hb(p.clipped_lulc_path, float(p.resampling_threshold), p.resampled_lulc_path,
                      resample_method='near',
                      gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS,
                      dst_ndv=hb.get_raster_info_hb(p.clipped_lulc_path)['nodata']
                      )


def reclassify_lulc():
    global p
    if p.lulc_categorization_table_path.endswith('.csv'):
        df = pd.read_csv(p.lulc_categorization_table_path, index_col=False)
    elif p.lulc_categorization_table_path.endswith('.xlsx'):
        df = pd.read_excel(p.lulc_categorization_table_path, index_col=False)
    elif p.lulc_categorization_table_path.endswith('.xls'):
        df = pd.read_excel(p.lulc_categorization_table_path, index_col=False)
    else:
        raise NameError('Unable to interpret ' + str(p.lulc_categorization_table_path) + '. Please save as CSV or XLSX.')

    value_map = {df['lulc_class_id'].values[c]: df['restoration_category_id'].values[c] for c in range(len(df['lulc_class_id'].values))}

    L.info('Reclassifying according to rules: ' + str(value_map))
    p.restoration_class_path = os.path.join(p.cur_dir, 'restoration_class.tif')
    hb.reclassify_raster((p.resampled_lulc_path, 1), value_map, p.restoration_class_path, 1,
                            255, values_required=True) # gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS

    p.forest_binary_path = os.path.join(p.cur_dir, 'forest_binary.tif')
    is_forest_map = {0: 0, 1: 1, 2: 0, 3: 0, 255: 0}
    hb.reclassify_raster((p.restoration_class_path, 1), is_forest_map, p.forest_binary_path, 1,
                            255, values_required=True) # gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS



    p.valid_mask_path = os.path.join(p.cur_dir, 'valid_mask.tif')
    hb.create_valid_mask_from_vector_path(p.area_of_interest_path, p.restoration_class_path, p.valid_mask_path)



    L.info('Reclassifying according to rules: ' + str(value_map))
    p.restoration_class_input_res_path = os.path.join(p.cur_dir, 'restoration_class_input_res.tif')
    hb.reclassify_raster((p.clipped_lulc_path, 1), value_map, p.restoration_class_input_res_path, 1,
                         255, values_required=True)  # gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS

    # Also reclassify the input LULC before resampling for masking the final results.
    p.is_restorable_path = os.path.join(p.cur_dir, 'is_restorable.tif')
    is_restorable_map = {0: 0, 1: 0, 2: 1, 3: 0, 255: 0}
    hb.reclassify_raster((p.restoration_class_input_res_path, 1), is_restorable_map, p.is_restorable_path, 1,
                         255, values_required=True)  # gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS

    p.valid_mask_input_res_path = os.path.join(p.cur_dir, 'valid_mask_input_res.tif')
    hb.create_valid_mask_from_vector_path(p.area_of_interest_path, p.restoration_class_input_res_path, p.valid_mask_input_res_path)


def calc_percent_of_overall_forest_cover_within_distance_threshold():
    global p
    convolution_edge_size = int(round(float(p.distance_threshold) / float(p.resampling_threshold), 0)) * 2 + 1
    convolution_array = np.zeros((convolution_edge_size, convolution_edge_size), dtype=np.int8)
    convolution_array = circle_from_ogrid(convolution_array, convolution_edge_size)

    convolution_path = os.path.join(p.cur_dir, 'convolution.tif')
    hb.save_array_as_geotiff(convolution_array, convolution_path, p.resampled_lulc_path, data_type=1, ndv=255,
                             n_cols_override=convolution_edge_size, n_rows_override=convolution_edge_size)

    p.percent_of_overall_forest_cover_within_threshold_path = os.path.join(p.cur_dir, 'percent_of_overall_forest_cover_within_threshold.tif')
    hb.convolve_2d((p.forest_binary_path, 1), (convolution_path, 1), p.percent_of_overall_forest_cover_within_threshold_path,
                   ignore_nodata=True, mask_nodata=True, normalize_kernel=False,
                   target_datatype=gdal.GDT_Float64,
                   target_nodata=255,
                   gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS,
                   n_threads=1, working_dir=None)


def calc_standardized_ecological_uncertainty():
    global p

    if not os.path.exists(p.output_dir):
        hb.create_directories(p.output_dir)

    p.standardized_ecological_uncertainty_analysis_res_path = os.path.join(p.cur_dir, 'standardized_ecological_uncertainty_analysis_res.tif')
    hb.raster_calculator_hb([(p.percent_of_overall_forest_cover_within_threshold_path, 1)], standardized_ecological_uncertainty,
                            p.standardized_ecological_uncertainty_analysis_res_path,
                            7, -9999.0, gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS)

    p.standardized_ecological_uncertainty_unmasked_path = os.path.join(p.cur_dir, 'standardized_ecological_uncertainty_unmasked.tif')
    hb.resample_to_match(p.standardized_ecological_uncertainty_analysis_res_path, p.clipped_lulc_path, p.standardized_ecological_uncertainty_unmasked_path)



    def mask_op(x, y):
        return np.where(x != 255, x * y, 255)
    p.standardized_ecological_uncertainty_unnormalized_path = os.path.join(p.cur_dir, 'standardized_ecological_uncertainty_unnormalized.tif')
    hb.raster_calculator_hb([(p.standardized_ecological_uncertainty_unmasked_path, 1),
                             (p.is_restorable_path, 1)],
                            mask_op, p.standardized_ecological_uncertainty_unnormalized_path,
                            7, -9999.0, gtiff_creation_options=hb.DEFAULT_GTIFF_CREATION_OPTIONS)

    x = hb.as_array(p.standardized_ecological_uncertainty_unnormalized_path)

    min = np.min(x)
    max = np.max(x)
    desired_max = 100.0
    scalar = desired_max / max

    o = np.where(x != -9999.0, x * scalar, -9999.0)
    x = None

    p.standardized_ecological_uncertainty_path = os.path.join(p.cur_dir, 'standardized_ecological_uncertainty.tif')
    hb.save_array_as_geotiff(o, p.standardized_ecological_uncertainty_path, p.standardized_ecological_uncertainty_unnormalized_path, data_type=7, ndv=-9999.0)

    r = hb.as_array(p.is_restorable_path)

    keys_where = np.where(r == 1)  # Not the NDV cause we're calculating deciles of the actually restorable land
    size = len(keys_where[0])
    output = np.ones(o.shape) * 255
    stride = int(size / 10.0)

    sorted_keys_1dim = o[keys_where].argsort(axis=None)
    sorted_keys = (keys_where[0][sorted_keys_1dim], keys_where[1][sorted_keys_1dim])
    for i in range(10):
        L.info('Calculating percentile ' + str((i + 1) * 10))

        output[sorted_keys[0][i * stride: (i + 1) * stride], sorted_keys[1][i * stride: (i + 1) * stride]] = i + 1
    output = output.reshape(o.shape)

    p.restoration_success_deciles_pre_final_mask_path = os.path.join(p.cur_dir, 'restoration_success_deciles_pre_final_mask.tif')
    hb.save_array_as_geotiff(output, p.restoration_success_deciles_pre_final_mask_path, p.standardized_ecological_uncertainty_unnormalized_path, data_type=1, ndv=255)

    ## NOTE FOR NEXT RELEASE: The following section was written to be memory safe and fast via raster_calculator_hb. However,
    ## This would make each tile in the calculation independent of others, which would incorrectly identify max value for as the LOCAL value not global.
    ## This resulted in tiling artifacts. And then again, the percentile calculaiton was messed up too. Thus, here, I reverted for time sake
    ## to non-memory safe numpy arrays.

    # def normalize_op(x):
    #     min = np.min(x)
    #     max = np.max(x)
    #     desired_max = 100.0
    #     scalar = desired_max / max

    #     return np.where(x != -9999.0, x * scalar, -9999.0)
    #
    # p.standardized_ecological_uncertainty_path = os.path.join(p.cur_dir, 'standardized_ecological_uncertainty.tif')
    # hb.raster_calculator_hb([(p.standardized_ecological_uncertainty_unnormalized_path, 1)], normalize_op, p.standardized_ecological_uncertainty_path, 7, -9999.0)
    #
    # def make_deciles(x, y):
    #
    #     keys_where = np.where(y == 1) # Not the NDV cause we're calculating deciles of the actually restorable land
    #     # keys_where = np.where(x != -9999.0)
    #     size = len(keys_where[0])
    #     output = np.ones(x.shape) * -9999.0
    #     stride = int(size / 10.0)
    #
    #     sorted_keys_1dim = x[keys_where].argsort(axis=None)
    #     sorted_keys = (keys_where[0][sorted_keys_1dim], keys_where[1][sorted_keys_1dim])
    #     for i in range(10):
    #         L.info('Calculating percentile ' + str((i + 1) * 10))
    #
    #         output[sorted_keys[0][i * stride: (i + 1) * stride], sorted_keys[1][i * stride: (i + 1) * stride]] = i + 1
    #     output = output.reshape(x.shape)
    #
    #     return output

    # p.restoration_success_deciles_pre_final_mask_path = os.path.join(p.cur_dir, 'restoration_success_deciles_pre_final_mask.tif')
    # hb.raster_calculator_hb([(p.standardized_ecological_uncertainty_path, 1),
    #                          (p.is_restorable_path, 1),
    #                          ], make_deciles, p.restoration_success_deciles_pre_final_mask_path, 1, 255)

    p.restoration_success_deciles_path = os.path.join(p.output_dir, 'restoration_success_deciles.tif')
    hb.set_ndv_by_mask_path(p.restoration_success_deciles_pre_final_mask_path, p.valid_mask_input_res_path, p.restoration_success_deciles_path)

    def cast_int(x):
        return np.byte(x)
    p.standardized_ecological_uncertainty_ints_pre_final_mask_path = os.path.join(p.cur_dir, 'standardized_ecological_uncertainty_ints_pre_final_mask.tif')
    hb.raster_calculator_hb([(p.standardized_ecological_uncertainty_path, 1)], cast_int, p.standardized_ecological_uncertainty_ints_pre_final_mask_path, 1, 255)

    p.standardized_ecological_uncertainty_ints_path = os.path.join(p.output_dir, 'standardized_ecological_uncertainty_percent.tif')
    hb.set_ndv_by_mask_path(p.standardized_ecological_uncertainty_ints_pre_final_mask_path, p.valid_mask_input_res_path, p.standardized_ecological_uncertainty_ints_path)


main = ''
if __name__ == '__main__':
    from hazelbean.ui import model, inputs
    import gofor_ui

    clip_lulc_task = p.add_task(clip_lulc)
    resample_lulc_task = p.add_task(resample_lulc)
    reclassify_lulc_task = p.add_task(reclassify_lulc)
    calc_percent_of_overall_forest_cover_within_distance_threshold_task = p.add_task(calc_percent_of_overall_forest_cover_within_distance_threshold)
    calc_standardized_ecological_uncertainty_task = p.add_task(calc_standardized_ecological_uncertainty)
    s = gofor_ui.GoforSplash()
    s.exec_()
    ui = gofor_ui.GoforUI(p)
    __version__ = '1.0.0'
    ui.links.setText(' | '.join((
        'GoFor: Ecological Restoration Uncertainty Assessment %s' % __version__,
        )))

    ui.run()
    EXITCODE = inputs.QT_APP.exec_()  # Enter the Qt application event loop. Without this line the UI will launch and then close.
