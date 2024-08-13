# Axial localisation using deep learning for SMLM
## Installation
Due to conflicting python dependencies, this module relies on two conda environments to function at this time.
Dependencies can be installed using conda:
```bash
git clone git@github.com:mb1069/smlm_z.git;
cd smlm_z;
conda env create --name smlm_z --file=environment.yml;
conda activate smlm_z;
pip install -e .;
```
## Usage
### 1. Pre-processing training data
Bead stacks should be placed into a common dir, using the directory format below. The script will look through this dir recursively for any .tif file. (Don't use 'combined' or 'slice' in the filename, as these are used to create intermediate/compiled files).

```
BEAD_STACK_DIR
|--dir1
|   |-file1.tif
|
|--dir2
    |-file2.tif
```

`smlm-prep-data` will run Picasso on a slice of each tif file, localising 2D beads and pre-processing them into a compiled file for model training.
```bash
conda activate smlm_z;
smlm-prep-data -z Z_STEP <BEAD_STACK_DIR>
```

Parameters `-qe -s -g -bl -a` are passed directly to Picasso's localisation; it is recommended to check these manually on a tif file before using them across all files if the default settings do not compile sensible training data. 

This will generate a new directory `<BEAD_STACK_DIR>/combined` with 3 files:
- `stacks.ome.tif`: a hyperstack of all beads extracted from the dataset
- `locs.hdf5`: localisation information of all the extracted beads
- `stacks_config.json` 

Use `--debug` to view debugging information for each extracted bead and rejected beads.
`--regen` will re-run Picasso - this is useful if you need to change the localisation settings.

### 2. Train model

Create a new directory containing the files from `<BEAD_STACK_DIR>/combined` (or provide all the paths to these files manually using arguments `-s <stacks.ome.tif> -l <locs.hdf5> -sc <stacks_config.json>` ).

Eg to train a model over a range of +- 1000nm and output the results into directory <OUTDIR>:
```bash
conda activate smlm_z;
smlm-train-model --zrange 1000 -o <OUT_DIR>
```

### 3. Localise experimental data
#### 3.1 X/Y localisation
Use Picasso to localise an SMLM experiment in X/Y; this is available from the following command:
```bash
picasso localize -b 15 <img.ome.tif>
```

#### 3.2 Z localisation
```bash
conda activate smlm_z;
smlm-localize -mo <OUT_DIR> -l <locs.hdf5> -s <spots.hdf5> -o <OUT_DIR>/out
```

#### 3.3 Z undrifting
Centroid-based undrifting is available using:
```bash
conda activate smlm_z;
smlm-undrift-z locs.hdf5
```
Alternately, use `picasso undrift` to undrift by RCC.

### 4. Post-processing scripts
Post-processing scripts exist for bacteria, nuclear pore or tubulin datasets; these can be used to generate X/Y X/Z Y/Z profiles and histograms for ROIs selected from Picasso's render interface.

To select objects in a dataset, run the command below, pick objects and save picked localisalations.
```bash
conda activate smlm_z;
picasso render
```
The saved localisation file can then be loaded into the following scripts:

```bash
conda activate smlm_z;
python publication/render_bac.py; for bacteria
python publication/render_nup.py; for nuclear pores
python publication/render_tub.py; for tubulin
```
