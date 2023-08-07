# A-GCL
---

Code developed and tested in Python 3.8.12 using PyTorch 1.10.0. Please refer to their official websites for installation and setup.

Some major requirements are given below

```python
numpy~=1.20.3
networkx~=2.7.1
torch~=1.10.0
scikit-learn~=1.0.2
scipy~=1.7.2
matplotlib~=3.5.0
torch-cluster~=1.6.0
torch-geometric~=2.0.4
torch-scatter~=2.0.9
torch-sparse~=0.6.13
torch-spline-conv~=1.2.1
```

## fMRI preprocessing and calculation of ALFF

---

After adjusting the data to the standarized format of BIDS, fMRIPrep was used to preprocess the data. 

Here is fMRIPrep script `fmriprep_Subj.sh`

```sh
bids_root_dir=$HOME/chenxiang/fmri_bids_data
subj=$sub
nthreads=4
container=docker #docker or singularity

#Begin:
export FS_LICENSE=$HOME/chenxiang/freesurfer/license.txt

#Run fmriprep
fmriprep-docker $bids_root_dir $bids_root_dir/ \
    participant \
    --participant-label $subj \
    --skip-bids-validation \
    --md-only-boilerplate \
    --fs-license-file $HOME/chenxiang/freesurfer/license.txt \
    --fs-no-reconall \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --nthreads $nthreads \
    --stop-on-first-crash \
    -w $HOME
```

Before calculate the ALFF, the AAL3 template needs to be registered to each subject.

Here is a registration script based on ANTs. `fmri_reg.py`

```python
import os
from glob import glob

data_path_1 = '/home/chenxiang/dataset/derivatives_fmriprep/fmriprep'
data_sub_name_1 = os.listdir(data_path_1)
data_sub_name_1.sort()

data_sub_path_1 = glob(data_path_1 + '/sub*')
data_sub_path_1.sort()

mni_1mm_path = '/home/chenxiang/data_temp/aal3/MNI152_T1_1mm_brain.nii.gz'
aal_mni_1mm_path = '/home/chenxiang/data_temp/aal3/AAL3_MNI_ID_1mm.nii'
aal2mni_lineartrans = '/home/chenxiang/data_temp/aal3/regAAL2MNI_ants_0GenericAffine.mat'
aal2mni_nlineartrans = '/home/chenxiang/data_temp/aal3/regAAL2MNI_ants_1Warp.nii.gz'

save_path = '/home/chenxiang/fmri_after_process'

for i_sub in range(0, len(data_sub_name_1)):
    fmri_prep_name = 'space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
    for root_1, dirs_1, files_1 in os.walk(data_sub_path_1[i_sub]):
        for file_ in files_1:
            data_sub_path_3 = os.path.join(root_1,file_)
            if fmri_prep_name in data_sub_path_3:
                sub_index_name = data_sub_path_3[89:103]
                save_path_name = save_path + '/' + sub_index_name
                if not os.path.exists(save_path_name):
                    os.makedirs(save_path_name)
                ref_path_name = data_sub_path_3[:-24] + 'boldref.nii.gz'
                if os.path.exists(ref_path_name):
                    print(sub_index_name)
                    outname_ants_1 = save_path_name + '/' + sub_index_name + '_regMNI2boldref_'
                    os.system(
                        'antsRegistrationSyNQuick.sh -d 3 -f {} -m {} -o {}'.format(ref_path_name, mni_1mm_path, outname_ants_1))
                    mni2bold_lineartrans = save_path_name + '/' + sub_index_name + '_regMNI2boldref_0GenericAffine.mat'
                    mni2bold_nlineartrans = save_path_name + '/' + sub_index_name + '_regMNI2boldref_1Warp.nii.gz'
                    outname_ants_2 = save_path_name + '/' + sub_index_name + '_aal3_atlas.nii'
                    os.system(
                        'antsApplyTransforms -d 3 -i {} -r {} -n NearestNeighbor -t {} -t {} -t {} -t {} -o {}'.format(aal_mni_1mm_path, ref_path_name, mni2bold_nlineartrans, mni2bold_lineartrans, aal2mni_nlineartrans, aal2mni_lineartrans, outname_ants_2))
                    os.system('rm *.nii.gz')
                    os.system('rm *.mat')

```

We used the DPABI_V7.0 to calculate the ALFF on MATLAB. The function in DPABI is `y_alff_falff.m`

```matlab
function [ALFFBrain, fALFFBrain, Header] = y_alff_falff(AllVolume,ASamplePeriod, HighCutoff, LowCutoff, AMaskFilename, AResultFilename, TemporalMask, ScrubbingMethod, Header, CUTNUMBER)
```

Here is an example about calculating one subject ALFF between 0.01Hz~0.08Hz

```matlab
[ALFF, fALFF, sub_0050002_header] = y_alff_falff(sub_0050002,sub_0050002_header.TR, 0.08, 0.01, sub_0050002_mask, sub_0050002_ALFF, [], ScrubbingMethod='cut', sub_0050002_header, 1)
```

## Datasets

---

We validated our proposed method on the Autism Brain Imaging Data Exchange I (ABIDE I), ABIDE II, and ADHD200 datasets and three different templates are used — AAL1(116), AAL3(166), Shen268(268).

In each template, the graphs are constructed with 3 bands of the amplitude of low-frequency fluctuation (ALFF) as node features and Pearson’s correlation coefficients (PCC) of the average fMRI time series in different brain regions as edge weights. Hence, the two main parts are processing as follow:

* Save the prepocessing `node features` and `edge weight` in the folder: ~/Data/raw. In our experiments, both of them are save in a ``.mat`` file. Specically, naming format of edge weight is: patient_id + template + type_of_edge, i.e.`sub_0050002_aal3_all_positive.mat`. and the corresponding node feature is `alff_sub-0050002.mat`.

📁：ABIDE

​	 |----------📁：raw

​	 --------------|----------📁：ASD_ADJ

​	 --------------|----------|------------ sub_0050002_aal3_all_positive.mat

​	 --------------|----------|------------ $\vdots$

​	 --------------|----------📁：ASD_NF

​	 --------------|----------|------------ alff_sub-0050002.mat

​	 --------------|----------|------------ $\vdots$

​    --------------|----------📁：HC_ADJ	
     
​    --------------|----------|------------ sub_0050002_aal3_all_positive.mat
     
​	 --------------|----------|------------ $\vdots$

​	 --------------|-----------📁：HC_NF			

​    --------------|----------|------------ alff_sub-0050002.mat
     
​	 --------------|----------|------------ $\vdots$

* Read the `node feature`:

  ```python
  nf = sio.loadmat(osp.join(path)) 
  x = nf['alff_value_cache']
  x = np.nan_to_num(x)
  x = torch.Tensor(x)
  ```

  Read the `edge weight` and transform it into `edge weight` and `edge index` in sparse format：

  ```python
  adj = sio.loadmat(osp.join(path))  
  edge_index = adj['corr_each_sub']
  
  edge_index = np.nan_to_num(edge_index)
  edge_index_temp = sp.coo_matrix(edge_index)
  edge_weight = torch.Tensor(edge_index_temp.data)
  
  edge_index = torch.Tensor(edge_index)   
  edge_index = edge_index.nonzero(as_tuple=False).t().contiguous()
  ```


## A-GCL Training

---

To perform self-supervised training on ABIDE using a 1-gpu machine, run:

```python
python agcl_ABIDE.py \ 
  --model_lr 0.0005 \
  --view_lr 0.0005 \
  --batch-size 32 \
  --num_gc_layers 2 \
  --emb_dim 32 \	
  --drop_ratio 0.3 \
  --reg_lambda 2.0 \
  --eval_interval 5 \
  --epochs 100
```

More details about arguments are concluded as follow:

```python
A-GCL ABIDE

optional arguments:
  -h, --help            show this help message and exit
  --model_lr MODEL_LR   Model Learning rate.
  --view_lr VIEW_LR     View Learning rate.
  --num_gc_layers NUM_GC_LAYERS
                        Number of GNN layers before pooling
  --pooling_type POOLING_TYPE
                        GNN Pooling Type Standard/Layerwise
  --emb_dim EMB_DIM     embedding dimension
  --mlp_edge_model_dim MLP_EDGE_MODEL_DIM
                        embedding dimension for MLP in view learner
  --batch_size BATCH_SIZE
                        batch size
  --drop_ratio DROP_RATIO
                        Dropout Ratio / Probability
  --epochs EPOCHS       Train Epochs
  --reg_lambda REG_LAMBDA
                        View Learner Edge Perturb Regularization Strength
  --eval_interval EVAL_INTERVAL 
  											eval epochs interval
  --downstream_classifier DOWNSTREAM_CLASSIFIER 
  											Downstream classifier is linear or non-linear
  --seed SEED
```

## A-GCL with dynamic memory bank(queue)

---

To perform self-supervised training with dynamic memory bank(queue) on ABIDE using a 1-gpu machine, run:

```python
python agcl_ABIDE_queue.py \ 
  --model_lr 0.0005 \
  --view_lr 0.0005 \
  --batch-size 32 \
  --num_gc_layers 2 \
  --emb_dim 32 \	
  --drop_ratio 0.3 \
  --reg_lambda 2.0 \
  --eval_interval 5 \
  --epochs 100 \
  --cr_lambda 0.4 \
  --memory_type 'queue' \
  --feature_type 'instance' \
  --max_length 256
```

More details about arguments are concluded as follow:

```python
A-GCL(queue) ABIDE

optional arguments:
  -h, --help            show this help message and exit
  --model_lr MODEL_LR   Model learning rate.
  --view_lr VIEW_LR     View learning rate.
  --num_gc_layers NUM_GC_LAYERS
                        Number of GNN layers before pooling
  --pooling_type POOLING_TYPE
                        GNN pooling type standard/Layerwise
  --emb_dim EMB_DIM     Embedding dimension
  --mlp_edge_model_dim MLP_EDGE_MODEL_DIM
                        Embedding dimension for MLP in view learner
  --batch_size BATCH_SIZE
                        Batch size
  --drop_ratio DROP_RATIO
                        Dropout ratio / Probability
  --epochs EPOCHS       Train epochs
  --reg_lambda REG_LAMBDA
                        View learner edge perturb regularization strength
  --eval_interval EVAL_INTERVAL 
  						Eval epochs interval
  --downstream_classifier DOWNSTREAM_CLASSIFIER 
  						Downstream classifier is linear or non-linear
  --cr_lambda
                        Regularization coefficients for loss of cross-batch memory bank 
  --memory_type         Type of memory bank
  --feature_type        Type of feature in memory bank
  --max_length          Max length of memory bank
  --seed SEED
```

If you have any questions about the settings or data pre-processing of A-GCL, please contact me through ``zsjxll@gmail.com``
If you find our work beneficial to your work, please cite our paper

```tex

```
