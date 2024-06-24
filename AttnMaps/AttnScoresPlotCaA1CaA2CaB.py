import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os
import cv2
import pandas as pd
import argparse
import shutil
import seaborn as sns
import json
import h5py
import glob
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy import stats

# ---------------------- Import and key info --------------------------------------
parser = argparse.ArgumentParser(description='Attn Maps')
parser.add_argument('--job_id', type=int, 
                    help='job_id determining the attn map to create according to a file list')
args = parser.parse_args()
id_file =  args.job_id


root = "/home/mathiane/LNENWork/LNEN_LCNEC_Molecular_barlow_twin/RoformerMILResA1A2B"
folder_tumoral_tiles = '/data/scratch/mathiane/Tiles_HE_all_samples_384_384_2'
folder_WSI_jpg = '/home/mathiane/LNENWork/FullSlidesToJpeg'
attn_map_folder = "attn_maps_formated"
im_size = 384
## Tab
pred = pd.read_csv(os.path.join(root,"predictions_all_test_folds.csv"))
pred_names =  [  ]
for ele in pred["preds"].values:
    if ele == 0:
        pred_names.append("CaA1")
    elif ele == 1 :
        pred_names.append("CaA2")
    else:
        pred_names.append("CaB")
        
pred=  pred.assign(pred_names = pred_names)


key_tab = pd.read_csv("/home/mathiane/LNENWork/LNEN_LCNEC_Molecular_barlow_twin/KEY_FILE_tneid_manuscript_id_match.csv", index_col=0)
## Outdir
outputdir = f'{root}/AttnMapPlot'
os.makedirs(f'{outputdir}',exist_ok=True)

# -------------- sample of interest -------------------------------------------------------
attn_map_flist = os.listdir(os.path.join(root,attn_map_folder ))
## To change arg line
attn_map_fname = attn_map_flist[id_file]
sample= attn_map_fname.split("_")[1].split(".csv")[0]
arch_sample = pred[pred["slide_id"]==sample]["labels_names"].values[0]
pred_arch_sample = pred[pred["slide_id"]==sample]["pred_names"].values[0]

print("sample: ", sample)
## FUll outfolder name
outfolder_arch = "Arch_" + arch_sample + "PredArch_" + pred_arch_sample
## tile folder 
tiles_folder = key_tab[key_tab["sample_id"]==sample]["tiles_folder"].values[0]

# ------------- load attention score -------------------------------------
attn_scores = pd.read_csv(os.path.join(root, attn_map_folder,  attn_map_fname))

# ------------- get max x and y tiles coords -----------------------------
sample_maxX_maxY = {}
path_main_TNE = folder_tumoral_tiles
sample = sample

sample_folder = os.path.join(path_main_TNE, tiles_folder)

xmax = 0
ymax = 0
for folder in os.listdir(sample_folder):
    tiles_p = os.path.join(path_main_TNE, tiles_folder, folder)
    for tiles_l in os.listdir(tiles_p):
        xmax_c = int(tiles_l.split('_')[1])
        ymax_c  = int(tiles_l.split('_')[2].split('.')[0])
        if xmax < xmax_c:
            xmax = xmax_c
        else:
            xmax = xmax
        if ymax < ymax_c:
            ymax = ymax_c
        else:
            ymax = ymax

sample_maxX_maxY[sample] = [xmax, ymax]
sample_maxX_maxY

# ----------  WSI overview fname ------------------------------------------
full_LNEN_WSI = folder_WSI_jpg
folder_name_full_size = ""
for f in  os.listdir(full_LNEN_WSI):

    if f.find(sample) != -1:
        folder_name_full_size = f
        break

# --------- create empty matrices --------------------------------------------
for k in sample_maxX_maxY.keys():
    w =  tuple(sample_maxX_maxY[k])[0] + im_size
    h = tuple(sample_maxX_maxY[k])[1] + im_size        
    seq = im_size
    W = len(list(range(1, w, seq)))
    H = len(list(range(1, h, seq)))
    
    mat_prob_ch0 =   np.empty((W*30, H*30))#-1
    mat_prob_ch0[:] =  np.NaN
    
    mat_prob_ch1 = np.empty((W*30, H*30))
    mat_prob_ch1[:] =  np.NaN
    
    mat_prob_ch2 = np.empty((W*30, H*30))
    mat_prob_ch2[:] =  np.NaN
        
df_test_pred_s = attn_scores

# ---------- Adaptative plot size --------------------------------------
if tuple(sample_maxX_maxY[k])[1] >= 20000:
    width_save_img = tuple(sample_maxX_maxY[k])[1] /2000
elif  tuple(sample_maxX_maxY[k])[1] >= 10000:
    width_save_img = tuple(sample_maxX_maxY[k])[1] /1000
elif  tuple(sample_maxX_maxY[k])[1] >= 5000:
    width_save_img = tuple(sample_maxX_maxY[k])[1] /200
else :
    width_save_img = tuple(sample_maxX_maxY[k])[1] /100
    
    
if tuple(sample_maxX_maxY[k])[0] >= 20000:
    height_save_img = tuple(sample_maxX_maxY[k])[0] /2000
elif  tuple(sample_maxX_maxY[k])[0] >= 10000:
    height_save_img = tuple(sample_maxX_maxY[k])[0] /1000
elif  tuple(sample_maxX_maxY[k])[0] >= 5000:
    height_save_img = tuple(sample_maxX_maxY[k])[0] /200
else:
    height_save_img = tuple(sample_maxX_maxY[k])[0] /100

# --------------- Create heatmap ------------------------------------------------
Path2Image = []
PredTumorNomal = []
loss_t = "attention_scores"
for k in sample_maxX_maxY.keys():
    if k in sample:  
        os.makedirs(f"{outputdir}/{outfolder_arch}/{k}", exist_ok=True)  
        for i in range(df_test_pred_s.shape[0]):
            x_ = int(df_test_pred_s.iloc[i,:]['x'])
            y_ = int(df_test_pred_s.iloc[i,:]['y'])
            
            Path2Image.append(df_test_pred_s.iloc[i,:]['img_id_c'])
            
            mat_prob_ch0[x_ // im_size * 30 :x_ // im_size *30 + 30 ,  y_ // im_size * 30 :y_ // im_size * 30 + 30 ]= df_test_pred_s.iloc[i,df_test_pred_s.columns.get_loc('attn_scores_ch0')]
            mat_prob_ch1[x_ // im_size * 30 :x_ // im_size *30 + 30 ,  y_ // im_size * 30 :y_ // im_size * 30 + 30 ]= df_test_pred_s.iloc[i,df_test_pred_s.columns.get_loc('attn_scores_ch1')]
            mat_prob_ch2[x_ // im_size * 30 :x_ // im_size *30 + 30 ,  y_ // im_size * 30 :y_ // im_size * 30 + 30 ]= df_test_pred_s.iloc[i,df_test_pred_s.columns.get_loc('attn_scores_ch2')]

        if folder_name_full_size != "":    
            if k.find('TNE') != -1:
                get_full_img = full_LNEN_WSI + "/" +folder_name_full_size
                print('get_full_img  ', get_full_img)

            im = cv2.imread(get_full_img)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            fig=plt.figure(1,figsize=(width_save_img,height_save_img))
            plt.imshow(im.astype('uint8'))
            plt.tick_params(left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False)
            print(outputdir, k+'_RoFormerMIL_','WSI_{}.png'.format(k))
            plt.title('WSI_{}'.format(k))
            fig.savefig(os.path.join(outputdir, outfolder_arch, k,'WSI_{}.png'.format(k)), dpi=fig.dpi)
            plt.close()

        #tiles heat map sans normalisation | CH0
        color_map = plt.cm.get_cmap('coolwarm')
        fig=plt.figure(2,figsize=(width_save_img,height_save_img))
     
        plt.matshow(mat_prob_ch0,  cmap=color_map,
                    interpolation='none',  fignum=2)
        mtitle = 'Attn Map Ch0 -  {} =  {}'.format(k, arch_sample )
        plt.tick_params(left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False)
        plt.title(mtitle)
        plt.colorbar()
        fig.savefig(os.path.join(outputdir, outfolder_arch,k, '{}Ch0_{}_{}.png'.format(loss_t, arch_sample, k, )), dpi=fig.dpi)
        plt.colorbar()
        plt.close()
                
       
        #tiles heat map sans normalisation | CH1
        color_map = plt.cm.get_cmap('coolwarm')
        fig=plt.figure(2,figsize=(width_save_img,height_save_img))
      
        plt.matshow(mat_prob_ch1,  cmap=color_map,
                    interpolation='none',  fignum=2)
        plt.tick_params(left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False)
        mtitle = 'Attn Map Ch1 -  {} =  {}'.format(k, arch_sample )
        plt.title(mtitle)
        plt.colorbar()
        fig.savefig(os.path.join(outputdir, outfolder_arch,k, '{}Ch1_{}_{}.png'.format(loss_t, arch_sample, k, )), dpi=fig.dpi)
        plt.colorbar()
        plt.close()
        
        
        
        #tiles heat map sans normalisation | CH1
        color_map = plt.cm.get_cmap('coolwarm')
        fig=plt.figure(2,figsize=(width_save_img,height_save_img))
        plt.matshow(mat_prob_ch2,  cmap=color_map,
                    interpolation='none',  fignum=2)
        plt.tick_params(left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False)
        mtitle = 'Attn Map Ch2 -  {} =  {}'.format(k, arch_sample )
        plt.title(mtitle)
        plt.colorbar()
        fig.savefig(os.path.join(outputdir, outfolder_arch,k, '{}Ch2_{}_{}.png'.format(loss_t, arch_sample, k, )), dpi=fig.dpi)
        plt.colorbar()
        plt.close()
        
# ----------- Extraction of key tules ------------------------------------------------
some_tiles_id = list(df_test_pred_s.sort_values(by=["attn_scores_ch0"], ascending=False)["img_id_c"].values[:20])
os.makedirs(os.path.join(outputdir, outfolder_arch  , k ,  'DiscrimiantTilesCh0'), exist_ok=True)
for tile  in some_tiles_id:
    t_path = os.path.join(folder_tumoral_tiles, tiles_folder, "accept", tiles_folder  + "_" + tile[8:] + ".jpg")
    t_path
    shutil.copy(t_path , os.path.join(outputdir, outfolder_arch , k ,  'DiscrimiantTilesCh0'))
    
some_tiles_id = list(df_test_pred_s.sort_values(by=["attn_scores_ch1"], ascending=False)["img_id_c"].values[:20])
os.makedirs(os.path.join(outputdir, outfolder_arch  , k ,  'DiscrimiantTilesCh1'), exist_ok=True)
for tile  in some_tiles_id:
    t_path = os.path.join(folder_tumoral_tiles, tiles_folder, "accept", tiles_folder  + "_" + tile[8:] + ".jpg")
    t_path
    shutil.copy(t_path , os.path.join(outputdir, outfolder_arch , k ,  'DiscrimiantTilesCh1'))
    
some_tiles_id = list(df_test_pred_s.sort_values(by=["attn_scores_ch2"], ascending=False)["img_id_c"].values[:20])
os.makedirs(os.path.join(outputdir, outfolder_arch  , k ,  'DiscrimiantTilesCh2'), exist_ok=True)
for tile  in some_tiles_id:
    t_path = os.path.join(folder_tumoral_tiles, tiles_folder, "accept", tiles_folder  + "_" + tile[8:] + ".jpg")
    t_path
    shutil.copy(t_path , os.path.join(outputdir, outfolder_arch , k ,  'DiscrimiantTilesCh2'))
    
    