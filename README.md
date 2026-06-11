# ESSD Supplementary Code and Model Files

This repository provides supplementary code, configuration files, utility modules, and trained model weights for the manuscript:

**Towards accurate daily 3D temperature and salinity reconstruction from remote sensing enhanced by explainable AI**

The repository is intended as supplementary material for the ESSD submission. The large reconstructed NetCDF dataset is not stored in this repository and is available from Zenodo.

## Repository contents

| File | Description |
|---|---|
| `trainRec_daily_alldepth.py` | Training script used for daily 3D temperature and salinity reconstruction experiments |
| `testRec_daily_ts_alldepth.py` | Inference and evaluation script used to generate reconstructed 3D fields |
| `Case11M.yaml` | Configuration file for the main reconstruction experiment |
| `tools.py` | Dataset loading and auxiliary functions |
| `visualization.py` | Visualization functions |
| `mdls.zip` | Model architecture modules |
| `utlts.zip` | Utility modules |
| `GLO_EF_temp3d11gradf_ADTOIAll_mask_Xnorm_epoch99.h5` | Trained model weights for 3D temperature reconstruction |
| `GLO_EF_salt3d11gradf_ADTOIAll_mask_Xnorm_epoch99.h5` | Trained model weights for 3D salinity reconstruction |

## Related dataset

The reconstructed temperature and salinity dataset is available on Zenodo:

**DOI: 10.5281/zenodo.20602639**

## Notes

The scripts were developed for local experiments and contain project-specific data loading settings. Users may need to adjust local data paths before running the scripts.

## Citation

Please cite the related manuscript and Zenodo dataset when using these supplementary files.
