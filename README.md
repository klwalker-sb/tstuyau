## tstuyau
TStuyau provides a robust, general-purpose framework for land-cover mapping including high-level image pre-processing, feature generation, and multi-factor model optimization. While it features highly specialized toolkits tailored for and small-scale farming classification and fire event mapping, tstuyau's core strength lies in its framework that allows for robust quality assessment and ability to account for many contributing factors when developing and selecting the best possible model. 

![alt](/images/tstuyau_graphic.jpg)

key features:
-- Raster pipeline for Landsat 5, 7, 8, 9 and Sentinel-2 processing, feature generation, and land cover model optimization 
-- Employs gridded structure, STAC imagery, and tools including dask, xarray, and geowombat to maximize parallel processing

### Install

```commandline
pip install git+https://github.com/klwalker-sb/tstuyau
```

### additional package requirements
- https://github.com/jgrss/geowombat
- https://github.com/jgrss/satsmooth
- https://github.com/jgrss/rastercrf
- https://github.com/jgrss/sacfei
- https://github.com/jgrss/pymorph3
- https://github.com/jgrss/eostac

### Usage
The features of tstuyau are maximized when chained directly to eostac, which applies atmospheric and BRDF corrections during STAC imagery ingestion. Utilizing eostac streamlines the pipeline by enforcing specific file structures and naming. Eostac also initiates a database that is picked up by tstuyau to track ingested imagery and processing steps per grid cell, ensuring seamless downstream feature generation and optimization.  

optional complimentary packages:
- hpchelpers (https://github.com/klwalker-sb/hpchelpers): tools for interacting with data on high performance computing environment 
- cultionet (https://github.com/jgrss/cultionet): semantic segmentation of cropland from satellite time series inputs using neural network model 

available commands = ['preprocess', 'move_nodata', 'mask', 'topo', 'fusion', 'reconstruct', 'reindex_si', 'reclassify_raster', 
                   'make_polygon_features', 'segment', 'make_ts_composite', 'make_var_stack', 'make_var_dataframe', 
                   'format_ptfeat_set', 'make_and_score_model', 'iterate_sample_model',  optimize_feature_model', 'iterate_all_model_components', 'classify_timestep', 'classify_CRF', 'mosaic', 'clean', 'assess', 
                   'compress', 'sample_timeseries', 'plot_timeseries', 'pre_post_df', 'pre_post_separability', 'dl_check', 'status', 'version', 'make_thumbnails', 'vectorize_seg_results', 'segmentation_accuracy', 
                   'prep_training_ts_for_segmentation', 'post_aggregation_filter', 'ts_filter']

### Use a custom configuration file

```commandline
tuyau preprocess --config custom_config.yaml
```

### Edit the default configuration settings

```commandline
tuyau preprocess --config-updates grids:[7,126] num_workers:4
```

### Pipeline
#### Image preprocessing ('no data' checks, co-registration, masks)

##### 1. Masking
Preliminary 'no data' check to remove images with no non-zeros values (which are common with gridded processing)

Cloud and shadow masks are applied via the following options: 
* Option 0: Use native cloud masks,
* Option 1: Download and apply S2cloudless masks from Google Earth Engine for Sentinel-2 images (recommended, and applied in eostac)
* Option 2: Use Conditional Random Fields trained on clouds, shadows, water, and clear land. -- requires additional input data

* Option 3: All methods can be followed by view-and-click interface method to remove any remaining low quality images.
```commandline
tuyau make_thumbnails --config-updates grids:[322]

or --script: scripts/bash_TStuyau_2qcc_thumbnails.sh--
```

followed by HPC helpers notebook 1b_ExploreData_RemoveCloudyImages
![alt](/images/qc1.png)

```commandline
tuyau clean --config-updates clean:treat_brdf:flag_cloudyX clean:xlist:cloud_images.csv

or --script: scripts/bash_TStuyau_2qcd_X_images_from_list.sh--
```

##### 2. Co-registration

Image-to-image co-registration with [AROSICS](https://gitext.gfz-potsdam.de/danschef/arosics).
  - The median of all available Landsat 8/9 images is used as a reference. Then, each Sentinel 2 image is co-registered to the Landsat reference as best possible, given that cloud cover may prevent co-registration of some images. The images that cannot be co-registered due to clouds or that have implausibly large shifts (set with 'coreg:max_shift' param) are marked with a 'X' in their filename prior to the extension and are excluded from subsequent analyses unless specified otherwise.  

```commandline
tuyau preprocess --config-updates grids[1,10]

or --script: scripts/bash_TStuyau_3_coreg.sh--
```

#### Sentinel 2/Landsat fusion to 10m

-- Option 1: Skip this step. All imagery will be processed at specified resolution. If smaller than native resolution, imagery is resampled with cubic convolution.

-- Option 2: Use a modified version of the [StarFM](https://ieeexplore.ieee.org/abstract/document/1661809) and the [improved phenology method](https://ieeexplore.ieee.org/abstract/document/7452606/) to fuse/sharpen Landsat 30m-->10m using Sentinel 2 references. See `gao_etal_2006` and `frantz_etal_2016`below.

#### Time series reconstruction

-- Option 1: dynamic temporal smoothing (DTS) method introduced in Graesser, Stanimirova, and Friedl (in prep A) to generate weekly time series.
-- Option 2: Other temporal smoothing methods including Savitzky-Golay, Whittaker, LOESS, Gaussian, and harmonic

```commandline
tuyau reconstruct

or --script: scripts/bash_TStuyau_4_ts_bywin.sh--
```
-- Option 3: No smoothing. For analyses involving abrupt or ephemeral change signals (e.g. burning,) direct image differencing will be more effective. In these cases, analyses will be more sensitive to noise. Different noise filters and quality control tools are provided, and the option 3 masking method is strongly recommended to remove problematic images.   

All options can be generated for individual bands or multi-band spectral indices. [See available indices here](#Spectral-indices). 


#### Time series plotting

> Command
 
```commandline
tuyau plot --config-updates grids:[<grid number>] plot:coords:"[-25.30028282,-55.10674438]" reconstruct:si:evi2 plot:out_path:/fig_path plot:start:'2013-07-01' plot:end:'2014-07-01'
```

#### Feature generation

- time-series features
- phenological features
- delta features and thresholding with noise control and normalization
- textural features (glcms)
- polygon-level features
- ancillary features (CHIRPS weather data, elevation, slope, jurisdictional data)

example of textural features:
![alt](/images/glcm.png)
```commandline
tuyau make_ts_composite

best via --script: bash_TStuyau_6_texture.sh
```

#### Vegetation or cropland object segmentation
-- Vector extraction and feature generation tools for segmantic segmentation outputs from cultionet 
![alt](/images/seg_images.jpg)

#### Feature visualization 
```commandline
tuyau make_ts_composite --config-updates grids:[<grid number>] feature_model:ts_type:raw feature_model:spec_indces:$SI classify:out_yrs:$MULTIYR feature_model:start_yr:$MODYR feature_model:si_vars:$SI_VARS feature_model:treat_out:$OUT calendar:first_mo:$STARTMO calendar:start_wet:$STARTWET calendar:end_wet:$ENDWET calendar:start_dry:$STARTDRY calendar:end_dry:$ENDDRY feature_model:use_pheno:$PHENO feature_model:spec_indices_pheno:$PHENOSIS feature_model:pheno_vars:$PHENOVARS feature_model:pheno_pad_days:$PHENOPAD
```
SI is the spectral index, entered as ```<spec_index>-<ts_type>-<ts_norm>``` (image optrions below)
SI_VARS are the features, entered as a list of ```<si_var>-<temp>``` (image options below)
The output is an image stack of length SI_VARS. Can also use 'Monthly' or 'Quarterly' in place of ```<temp>``` to create one band for each month/quarter.

![alt](/images/tstuyau_features.png)

![alt](/images/tstuyau_model_parts.png)

#### Model optimization

![alt](/images/tstuyau_model_optimization.png)

#### Classification architecture

Options to classify a single timestep at a time or multiple timesteps at once with change as an integral model component. 

-- For single timestep: 
Current options include Random Forest and Gradient Boosting

-- For multi year:
We use sequential land cover mapping with Conditional Random Fields, introduced in Graesser *et al.* (in prep B).

> Command

##### Final classification for wall-to-wall map

```commandline
tuyau classify_timestep

or script: scripts/bash_TStuyau_8_classify_image.sh
```

### Final contextual filters

#### contextual filters for smallholder agriculture:
 - changes crop pixels within segmented polygons to majority crop
 -  identifies crop pixels in fields < 2 ha as smallholder if in areas away from large fields
 -  converts pixels classified as smallholder mixed along edges of large fields as crop edge (not smallholder)
 -  cleans up other common misclassification of mixed vegetation as smallholder crop, such as borders between grasslands and high vegetation
 -  converts all low mixed veg that is not identified as smallholder crop to crop edge if it borders crop pixels and mixed grass otherwise 

scripts/bash_TStuyau_9b_filter_context.sh
![alt](/images/post_filters.png)

#### temporal filters:
- gets majority vote across time series for classes not expected to fluctuate from year to year (e.g. forest type)
- refines classification of some classes based on future observations (e.g. shrub crops to young tree plantation if observed as mature tree plantation later)
- downgrades classification for illogically abrupt changes (tree changed to shrub if grass -> tree occurs in single year)
- can correct for many predefined illogical sequences selected from list
- option to apply regional filter to any correction
   
scripts/bash_TStuyau_9c_filter_ts.sh

## Spectral indices
| Index | Formula | Uses | Source |
|---|---|---|---|
| NDVI | $\frac{NIR-Red}{NIR+Red}$ | Assessing vegetation health and density. | Rouse et al. (1974) |
| kNDVI | $tanh\left(\left(\frac{NIR-Red}{NIR+Red}\right)^2\right)$ | More sensitive to range of landscapes -- dense and sparse.| Camps-Valls et al. (2021) |
| GCVI | $\frac{NIR}{Green} - 1$ | | |
| EVI | $G \times \frac{NIR - Red}{NIR + C_1 \times Red - C_2 \times Blue + L}$ | Improved sensitivity in high biomass regions. | Huete et al. (2002) |
| EVI2| $2.5 \times \frac{NIR - Red}{NIR + 2.4 \times Red + 1}$ |High-biomass monitoring for sensors lacking a Blue band. | Jiang et al. (2008) |
| SAVI | $$SAVI = \left(\frac{NIR - Red}{NIR + Red + L}\right) \times (1 + L)$$ | Correcting soil brightness in low-cover areas. | Huete (1988) |
| MSAVI | $$\frac{(2 \times NIR + 1) - \sqrt{(2 \times NIR + 1)^2 - 8 \times (NIR - Red)}}{2}$$ | Version of SAVI that eliminates the need to choose an L-factor | Qi (1994) |
| WI | $` \begin{cases} 0.001 & \text{\small if } (SWIR1+Red) > 0.5 \\ 1 - \frac{SWIR1 + Red}{0.5} & \text{\small otherwise} \end{cases} `$ | To seperate woody growth from herbaceous. | Lehmann et al (2013)|
| NDMI | $\frac{NIR-SWIR1}{NIR+SWIR1}$ | Moisture Index | Gao (1996)
| NBR | $\frac{NIR-SWIR2}{NIR+SWIR2}$ | Identification of burned areas amid healthy vegetation | |
| NBR2| $\frac{SWIR1-SWIR2}{SWIR1+SWIR2}$ |Identification of burned areas in areas without photosynthetically active vegetation | |
| BAI | $$\frac{1}{(0.1 - \text{Red})^2 + (0.06 - \text{NIR})^2}$$ | Identification of burned areas|Chuvieco et al. (2002)|
| BAIM | $$\frac{1}{(0.06 - \text{NIR})^2 + (0.215 - \text{SWIR})^2}$$ | Improved separability of charcoal vs. non-fire dark covers||
| dNBR | $NBR_{prefire} - NBR_{postfire}$ | To measure burn intensity | |
| rdNBR | $\frac{dNBR}{\sqrt{\|NBR_{prefire}\|}}$ | dNBR normalized by pre-fire vegetation cover | Miller and Thode (2007) |



<!--  or something like this (remove first and last lines)
 note: Use $ for inline or $$ for a larger, centered block
<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Formula</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>kNDVI</td>
      <td>$\tanh\left(\left(\frac{NIR-Red}{NIR+Red}\right)^2\right)$</td>
      <td>Camps-Valls et al. (2021)/td>
    </tr>
    <tr>
      <td colspan="3" align="center">
       Kernel Normalized Difference Vegetation Index. For..
      </td>
    </tr>
  </tbody>
    <tr>
      <td>rdNBR</td>
      <td>$\frac{dNBR}{\sqrt{\|NBR_{prefire}\|}}$</td>
      <td> Miller and Thode (2007) /td>
    </tr>
    <tr>
      <td colspan="3" align="center">
       relative delta NBR..  
      </td>
    </tr>
</table>
-->




## References

```bibtex
@article{gao_etal_2006,
  title={On the blending of the Landsat and MODIS surface reflectance: Predicting daily Landsat surface reflectance},
  author={Gao, Feng and Masek, Jeff and Schwaller, Matt and Hall, Forrest},
  journal={IEEE Transactions on Geoscience and Remote sensing},
  volume={44},
  number={8},
  pages={2207--2218},
  year={2006},
  publisher={IEEE}
}

@article{frantz_etal_2016,
  title={Improving the spatial resolution of land surface phenology by fusing medium-and coarse-resolution inputs},
  author={Frantz, David and Stellmes, Marion and R{\"o}der, Achim and Udelhoven, Thomas and Mader, Sebastian and Hill, Joachim},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={54},
  number={7},
  pages={4153--4164},
  year={2016},
  publisher={IEEE}
}

@article{graesser_ramankutty_2017,
  title={Detection of cropland field parcels from Landsat imagery},
  author={Graesser, Jordan and Ramankutty, Navin},
  journal={Remote Sensing of Environment},
  volume={201},
  pages={165--180},
  year={2017},
  publisher={Elsevier}
}

@article{graesser_etal_inprep_a,
  author={Graesser, Jordan and Stanimirova, Radost and Friedl, Mark},
  title={Reconstruction of satellite time series with a dynamic smoother},
  journal={xxxx},
  year={xxxx}
}

@article{graesser_etal_inprep_b,
  author={Graesser, Jordan and Stanimirova, Radost and Friedl, Mark and Copat\'{i}, Esteban and Volante, Jos\'{e} and Veron, Santiago and Banchero, Santiago and Elena, Hernan},
  title={Generating annual land cover estimates with sequential models and Landsat time series},
  journal={xxxx},
  year={xxxx}
}
```
