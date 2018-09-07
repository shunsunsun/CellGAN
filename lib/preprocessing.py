import fcm
import os
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import pickle

# settings

markers_of_interest = ['CD3', 'CD45', 'CD4', 'CD20', 'CD33', 'CD123', 'CD14', 'IgM', 'HLA-DR', 'CD7']
regex_well = '_C12.fcs'
inhibitor_used = 'Crassin'

dir_data = os.path.join(sys.path[0], inhibitor_used)

def ftrans(x, c):
    return np.arcsinh(1./c * x)

# load data
list_files = os.listdir(dir_data)

p = re.compile(regex_well)
files_of_interest = list()
cell_type = list()

for file in list_files:
    if bool(p.search(file)):
        files_of_interest.append(file)
        ctype = file.split('_')[1]
        cell_type.append(ctype)

#print("Regex well used {}\n".format(regex_well))        
#print('Number of files of interest is {}\n'.format(len(files_of_interest)))


# load data
data = list()
label = list()
celltype_added = 0
#counts = {}
#total = 0

#for ctype in cell_type:
#    counts[ctype] = 0

for i, file in enumerate(files_of_interest):
    
    temp_fsc = fcm.loadFCS(os.path.join(dir_data, file))
    temp_idx_marker_of_interest = np.asarray([np.where(np.asarray(temp_fsc.channels) == x)[0] for x in markers_of_interest])
	
    ncells = np.array(temp_fsc).shape[0]
        
    if ncells >= 30:
    
        data.append(ftrans(np.squeeze(np.array(temp_fsc)[:, temp_idx_marker_of_interest]), 5))
        #counts[cell_type[i]] += data[i].shape[0]
	label.append([celltype_added]*data[celltype_added].shape[0])
        celltype_added += 1
        #total += ncells
    else:
        continue

print("The number of celltypes added {}".format(celltype_added))
data = np.vstack(data)
label = np.concatenate(label)

real_data_dir = '/'.join(sys.path[0].split('/')[:-1]) + '/CellGan_tensorflow/Real Data'
output_dir = os.path.join(real_data_dir, inhibitor_used)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file_label = os.path.join(output_dir, regex_well.split('.')[0]) + '_label.pkl' 
output_file = os.path.join(output_dir, regex_well.split('.')[0]) + '.pkl'

if not os.path.exists(output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    with open(output_file_label, "w") as f:
        pickle.dump(label, f)

plotting = False
# plot pca 
if plotting:

    pca_transform = PCA(n_components=2).fit_transform(data)
    plt.figure()
    plt.scatter(pca_transform[:,0], pca_transform[:,1], c=label, label=label.astype(str))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(sys.path[0] + '/test.png')


# plot pca 
#tsne_transform = TSNE(n_components=2).fit_transform(data)
#plt.figure()
#plt.scatter(tsne_transform[:,0], tsne_transform[:,1], c=label, label=label.astype(str))
#plt.xlabel('dim 1')
#plt.ylabel('dim 2')
#plt.legend()
#plt.tight_layout()
#plt.savefig(sys.path[0] + 'test_tsne.png')
