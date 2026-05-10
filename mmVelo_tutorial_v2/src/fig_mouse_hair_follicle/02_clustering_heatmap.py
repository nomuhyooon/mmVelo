import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
from scipy.io import mmwrite, mmread
import seaborn as sns
import scipy

np.random.seed(42)


# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")
pseudotime = pd.read_csv(dir_path+"/pseudotime.tsv", sep="\t", header=None)[0].to_numpy()
adata_r.obs["pseudotime"] = pseudotime
refined_clusters = pd.read_csv(dir_path+"/refined_clusters.tsv", sep="\t", header=None)[0].to_numpy()
adata_r.obs["ref_clusters"] = refined_clusters

#np.savetxt(dir_path+"/atac_reconstructed.txt", adata_a.layers["a_raw"].toarray())
#np.savetxt(dir_path+"/atac_velocity.txt", adata_a.layers["dadt"].toarray())
#np.savetxt(dir_path+"/obs_names.txt", adata_r.obs_names.to_numpy(), fmt='%s')
adata_r.obs_names.to_numpy()

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

# cluster-wise pseudotime
# maybe this should be done with TAC-subclustering i.e. IRS-TAC, HS-CC TAC
#set(adata_r.obs["clusters"])
#cell_clusters = ['TAC', 'Inner Root Sheath', 'Hair Shaft-Cuticle/Cortex', 'Medulla']
set(adata_r.obs["ref_clusters"])
cell_clusters = ['TAC', 'IRS-TAC', 'Inner Root Sheath', 'HS-TAC', 'Hair Shaft-Cuticle/Cortex', 'Medulla']
sorted_cell_barcode = []
for cell_cluster in cell_clusters:
    cell_cluster
    adata_r_cluster = adata_r[adata_r.obs["ref_clusters"] == cell_cluster, :]
    sorted_cells = list(adata_r_cluster.obs_names[np.argsort(adata_r_cluster.obs["pseudotime"])])
    sorted_cell_barcode.append(sorted_cells)
sorted_cell_barcode = sum(sorted_cell_barcode,[])
adata_r = adata_r[sorted_cell_barcode, :]
adata_a = adata_a[sorted_cell_barcode, :]

adata_r[adata_r.obs["clusters"] == "Hair Shaft-Cuticle/Cortex", :].obs["pseudotime"].max()
adata_r[adata_r.obs["clusters"] == "Medulla", :].obs["pseudotime"].max()
adata_r[adata_r.obs["clusters"] == "Inner Root Sheath", :].obs["pseudotime"].max()

a_raw = adata_a.layers["a_raw"].toarray()
dadt = adata_a.layers["dadt"].toarray()

"""
# not ordering by clusters
a_raw = adata_a.layers["a_raw"].toarray()[np.argsort(adata_r.obs["pseudotime"]), :]
dadt = adata_a.layers["dadt"].toarray()[np.argsort(adata_r.obs["pseudotime"]), :]
"""

var_name = adata_a.var_names.to_numpy()
dadt = dadt / np.std(dadt, axis=0)
a_raw = scipy.stats.zscore(a_raw)

# clustering
resolution = 1.0
adata_bin = ad.AnnData(X=dadt.T)
adata_bin.layers["rec_x"] = a_raw.T
adata_bin.obs_names = var_name
sc.pp.neighbors(adata_bin, n_neighbors=30, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.leiden(adata_bin, resolution=resolution)
sc.tl.umap(adata_bin)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata_bin, color="leiden")
fig.tight_layout()
plt.savefig(dir_path + "/umap_leiden_resolution_{}.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata_bin, color="leiden")
plt.gca().axis("off")
plt.title("")
fig.tight_layout()
plt.savefig(dir_path + "/umap_leiden_resolution_{}_blank.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()

def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap('tab20')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    return cluster_colors

clusters = adata_bin.obs["leiden"]
num_clusters = len(adata_bin.obs["leiden"].cat.categories)
cluster_colors = generate_cluster_colors(num_clusters)

sorted_dadt = dadt[:, np.argsort(clusters)]
sorted_var_name = var_name[np.argsort(clusters)]


sorted_dadt = sorted_dadt.T
sorted_x_raw = a_raw[:, np.argsort(clusters)]
sorted_x_raw = sorted_x_raw.T
clusters = clusters[np.argsort(clusters)]
row_colors = [cluster_colors[int(i)] for i in clusters]


# set columns colors
col_annots = [list(adata_r.obs["ref_clusters"]), list(adata_r.obs["pseudotime"])]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["clusters", "pseudotime"])

sc.pl.umap(adata_r, color="ref_clusters")
plt.close("all")
clusters_labels = col_annots.get_level_values("clusters")
# Categories (4, object): ['Hair Shaft-Cuticle/Cortex', 'Inner Root Sheath', 'Medulla', 'TAC']

#clusters_colors = [adata_r.uns["clusters_colors"][3], 
#                   adata_r.uns["clusters_colors"][0],
#                   adata_r.uns["clusters_colors"][1],
#                   adata_r.uns["clusters_colors"][2]]
#clusters_pal = [matplotlib.colors.to_rgb(hex) for hex in clusters_colors]
#clusters_pal = [matplotlib.colors.to_rgb(hex) for hex in adata_r.uns["ref_clusters_colors"][[5, 2, 3, 0, 1, 4]]]
clusters_colors = [adata_r.uns["ref_clusters_colors"][5], 
                   adata_r.uns["ref_clusters_colors"][2],
                   adata_r.uns["ref_clusters_colors"][3],
                   adata_r.uns["ref_clusters_colors"][0],
                   adata_r.uns["ref_clusters_colors"][1],
                   adata_r.uns["ref_clusters_colors"][4]
                   ]
clusters_pal = [matplotlib.colors.to_rgb(hex) for hex in clusters_colors]
clusters_lut = dict(zip(map(str, clusters_labels.unique()), clusters_pal))
clusters_colors = pd.Series(clusters_labels, index=col_annots).map(clusters_lut) 

pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([clusters_colors, pdt_colors], axis=1)


sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_x_raw, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_x.png".format(resolution), bbox_inches='tight')
plt.close()

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dadt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()

"""
# sort peaks according to pseudotime
cluster_order = [4, 6, 5, 0, 1, 2, 7, 3, 8]
peak_order = []

for i in cluster_order:
    peak_order.append(list(np.where(adata_bin.obs["leiden"].to_numpy() == str(i))[0]))
peak_order = sum(peak_order, [])

row_colors_sorted = adata_bin.obs["leiden"][peak_order]
row_colors_sorted = [cluster_colors[int(i)] for i in row_colors_sorted]

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(dadt[:, peak_order].T, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors_sorted, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_sorted.png".format(resolution), bbox_inches='tight')
plt.close("all")


sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(a_raw[:, peak_order].T, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors_sorted, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_x_sorted.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()

# for fig
sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(dadt[:, peak_order].T, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors_sorted, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_sorted_blank.png".format(resolution), bbox_inches='tight')
plt.close("all")

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(a_raw[:, peak_order].T, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors_sorted, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_x_sorted_blank.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()
"""



# save adata to perform motif enrichment analysis
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
adata_bin.write_loom(dir_path+"/adata_dadt_cluster.loom", write_obsm_varm=True)


# read adata
## change environment to scenicplus
## also load adata_r for tf 
import scanpy as sc
import pycistarget
import pyranges as pr
import pickle
import statsmodels.api as sm

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
adata_peak = sc.read_loom(dir_path+"/adata_dadt_cluster.loom")
adata_peak.obs_names = adata_peak.obs["obs_names"]
peaks = list(adata_peak.obs_names)

bg_peaks = pd.read_csv("/home/nomura/Proj/mmvelo/data/share_seq/unfiltered_peak_set.bed", sep="\t", header=None,
            names = ["Chromosome", "Start", "End"])
bg_peaks["peak"] = bg_peaks["Chromosome"].astype("str") + ":" + bg_peaks["Start"].astype("str") + "-" + bg_peaks["End"].astype("str")
bg_peaks = list(set(bg_peaks["peak"]).difference(set(adata_peak.obs_names)))

region_sets = dict()
sets_keys = ["for_py_ctx"]
contrasts = list() # for DEM
for set_key in sets_keys:
    #region_sets[set_key] = dict()
    region_sets = dict()
    for clst in set(adata_peak.obs["leiden"]):
        clst_peaks = adata_peak.obs_names[adata_peak.obs["leiden"] == clst]
        clst_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in clst_peaks],
                        "Start": [int(peak.split(':')[1].split('-')[0]) for peak in clst_peaks],
                        "End": [int(peak.split(':')[1].split('-')[1]) for peak in clst_peaks]})
        key = f'leiden_{clst}'
        #region_sets[set_key][key] = clst_peaks
        region_sets[key] = clst_peaks
        contrasts.append([[key], ["background"]])
    bg_peak = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in bg_peaks],
                    "Start": [int(peak.split(':')[1].split('-')[0]) for peak in bg_peaks],
                    "End": [int(peak.split(':')[1].split('-')[1]) for peak in bg_peaks]})
    
    region_sets_dem = region_sets.copy()
    region_sets_dem["background"] = bg_peak
    #region_sets[set_key]["background"] = bg_peak
    


from pycistarget.motif_enrichment_cistarget import *
from pycistarget.motif_enrichment_dem import *
from scenicplus.wrappers.run_pycistarget import run_pycistarget
###

region_sets_dem.keys()

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
save_dir = os.path.join(dir_path, "scenicplus_sep")
os.mkdir(save_dir)

temp_dir = os.getcwd() + "/temp"
os.mkdir(temp_dir)

# the same parameter as run_pycistarget function in scenicplus
DEM_dict = DEM(dem_db = '/home/nomura/Proj/mmvelo/scenicplus/mm10_screen_v10_clust.regions_vs_motifs.scores.feather',
    region_sets = region_sets_dem,
    specie = 'mus_musculus',
    contrasts = contrasts,
    name = 'DEM',
    fraction_overlap = 0.4,
    max_bg_regions = 1000,
    log2fc_thr = 0.5,
    mean_fg_thr = 0,
    motif_hit_thr = 3.0,
    adjpval_thr = 0.05,
    n_cpu = 8,
    annotation_version = 'v10nr_clust',
    path_to_motif_annotations = '/home/nomura/Proj/mmvelo/pycistarget/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl',
    motif_annotation = ['Direct_annot', 'Orthology_annot'],
    tmp_dir = temp_dir
    )

import pickle
with open(save_dir + "/DEM_dict.pkl", 'wb') as f:
    pickle.dump(DEM_dict, f)

infile = open(save_dir + "/DEM_dict.pkl", 'rb')
DEM_dict = pickle.load(infile)
infile.close()

cistarget_dict = run_cistarget(ctx_db = '/home/nomura/Proj/mmvelo/pycistarget/mm10_screen_v10_clust.regions_vs_motifs.rankings.feather',
                                                      region_sets = region_sets,
                                                      specie = 'mus_musculus',
                                                      auc_threshold = 0.005,
                                                      nes_threshold = 3.0,
                                                      rank_threshold = 0.05,
                                                      annotation = ['Direct_annot', 'Orthology_annot'],
                                                      annotation_version = 'v10nr_clust',
                                                      path_to_motif_annotations = '/home/nomura/Proj/mmvelo/pycistarget/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl',
                                                      n_cpu = 8,
                                                      #_temp_dir='/scratch/leuven/313/vsc31305/ray_spill'
                                                      )

with open(save_dir + "/cistarget_dict.pkl", 'wb') as f:
    pickle.dump(cistarget_dict, f)

infile = open(save_dir + "/cistarget_dict.pkl", 'rb')
cistarget_dict = pickle.load(infile)
infile.close()


cistarget_dict
DEM_dict.motif_enrichment

for clst in set(adata_peak.obs["leiden"]):
    clst_key_ctx = "leiden_" + clst
    clst_key_dem = clst_key_ctx + "_VS_background"
    
    # CTX
    tf_motifs_ctx = cistarget_dict[clst_key_ctx].motif_enrichment
    tf_lists_ctx = list(tf_motifs_ctx["Direct_annot"]) + list(tf_motifs_ctx["Orthology_annot"])
    
    tf_list_ctx = []
    for i in range(len(tf_lists_ctx)):
        if isinstance(tf_lists_ctx[i], float):
            continue
        tf_lists_ctx[i] = tf_lists_ctx[i].split(", ")
        tf_list_ctx.append(tf_lists_ctx[i])
    tf_list_ctx = sum(tf_list_ctx, [])
    
    # DEM
    tf_motifs_dem = DEM_dict.motif_enrichment[clst_key_dem]
    if len(tf_motifs_dem) == 0:
        tf_list_dem = list()
    else:
        tf_lists_dem = list(tf_motifs_dem["Direct_annot"]) + list(tf_motifs_dem["Orthology_annot"])
        tf_list_dem = []
        for i in range(len(tf_lists_dem)):
            if isinstance(tf_lists_dem[i], float):
                continue
            tf_lists_dem[i] = tf_lists_dem[i].split(", ")
            tf_list_dem.append(tf_lists_dem[i])
        tf_list_dem = sum(tf_list_dem, [])
    
    tf_list_clst = tf_list_ctx + tf_list_dem
    
    exists_tf_list = []
    for tf in tf_list_clst:
        if tf in adata_r.var_names:
            exists_tf_list.append(tf)
    exists_tf_list = list(set(exists_tf_list))
    
    with open(save_dir + "/exist_tf_motifcluster_{}.txt".format(clst), "w+") as output:
        output.write(str(exists_tf_list))





"""
######
run_pycistarget(
    region_sets = region_sets,
    species = 'mus_musculus',
    save_path = save_dir,
    ctx_db_path = "/home/nomura/Proj/mmvelo/pycistarget/mm10_screen_v10_clust.regions_vs_motifs.rankings.feather",
    dem_db_path = "/home/nomura/Proj/mmvelo/scenicplus/mm10_screen_v10_clust.regions_vs_motifs.scores.feather",
    path_to_motif_annotations = "/home/nomura/Proj/mmvelo/pycistarget/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl",
    run_without_promoters = False,
    n_cpu = 8,
    annotation_version = 'v10nr_clust',
    dem_max_bg_regions = 30000,
    #contrasts = contrasts
    )

import dill
menr = dill.load(open(dir_path + "/scenicplus/menr.pkl", "rb"))

from scenicplus.cistromes import *

def _signatures_to_iter(menr):
    for x in menr.keys():
        if isinstance(menr[x], pycistarget.motif_enrichment_dem.DEM):
            for y in menr[x].cistromes['Region_set'].keys():
                for z in menr[x].cistromes['Region_set'][y]:
                    yield x, y, z, menr[x].cistromes['Region_set'][y][z]
        elif isinstance(menr[x], dict):
            for y in menr[x].keys():
                if not isinstance(menr[x][y], pycistarget.motif_enrichment_cistarget.cisTarget):
                    raise ValueError(f'Only motif enrichment results from pycistarget or DEM are allowed, not {type(menr[x][y])}')
                for z in menr[x][y].cistromes['Region_set']:
                    yield x, y, z, menr[x][y].cistromes['Region_set'][z]
        else:
            raise ValueError(f'Only motif enrichment results from pycistarget or DEM are allowed, not {type(menr[x])}')
        
def _get_signatures_as_dict(i):
    return {z+'__'+x+'__'+y: regions for x, y, z, regions in i}

# Get signatures from Homer/Cistarget outputs
signatures = _get_signatures_as_dict(_signatures_to_iter(menr))

#split direct and indirect signatures
signatures_direct = {x: signatures[x] for x in signatures.keys() if not 'extended' in x}
signatures_extend = {x: signatures[x] for x in signatures.keys() if     'extended' in x}

#Remove empty signature
signatures_direct = {k:v for k,v in signatures_direct.items() if v}
signatures_extend = {k:v for k,v in signatures_extend.items() if v}

if len(signatures_direct.keys()) == 0 and len(signatures_extend.keys()) == 0:
    raise ValueError("No cistromes found! Make sure that the motif enrichment results look good!")

#merge regions by TF name
def _merge_dict_of_signatures(d, suffix = ''):
    arr_keys_signatures = np.array(list(d.keys()))
    grouper = Groupby([x.split('_')[0] for x in arr_keys_signatures])
    merged_signatures = {}
    for TF, idx in zip(grouper.keys, grouper.indices):
        merged_signatures[TF + suffix] = pr.PyRanges(
            region_names_to_coordinates(set(flatten_list([d[x] for x in arr_keys_signatures[idx]]))))
    return merged_signatures

if len(signatures_direct.keys()) > 0:
    merged_signatures_direct = _merge_dict_of_signatures(signatures_direct, suffix = '')
if len(signatures_extend.keys()) > 0:
    merged_signatures_extend = _merge_dict_of_signatures(signatures_extend, suffix = '_extended')

#overlap regions with scplus_regions
regions = set(adata_peak.obs_names)
pr_regions = pr.PyRanges(region_names_to_coordinates(regions))
regions_to_overlap = pr_regions

def _overlap_if_necessary(d, test_regions, regions_to_overlap):
    d_overlap = {}
    for k in d.keys():
        s_query_regions = set(coord_to_region_names(d[k]))
        #if the signature regions are already in the scplus_obj coordinate system, do nothing, otherwise overlap
        if len(s_query_regions & test_regions) != len(s_query_regions):
            signature_regions = target_to_overlapping_query(regions_to_overlap, d[k])
        else:
            signature_regions = d[k]
        if len(signature_regions) != 0:
            d_overlap[k] = signature_regions
    return d_overlap

if len(signatures_direct.keys()) > 0:
    merged_signatures_direct = _overlap_if_necessary(merged_signatures_direct, regions, regions_to_overlap)
if len(signatures_extend.keys()) > 0:
    merged_signatures_extend = _overlap_if_necessary(merged_signatures_extend, regions, regions_to_overlap)
    
# Sort alphabetically
if len(signatures_direct.keys()) > 0:
    merged_signatures_direct = dict(
        sorted(merged_signatures_direct.items(), key=lambda x: x[0].lower()))
if len(signatures_extend.keys()) > 0:
    merged_signatures_extend = dict(
        sorted(merged_signatures_extend.items(), key=lambda x: x[0].lower()))

# Combine
if len(signatures_direct.keys()) > 0 and len(signatures_extend.keys()) > 0:
    merged_signatures = {**merged_signatures_direct,
                            **merged_signatures_extend}
elif len(signatures_direct.keys()) > 0 and len(signatures_extend.keys()) == 0:
    merged_signatures = merged_signatures_direct
elif len(signatures_extend.keys()) > 0 and len(signatures_direct.keys()) == 0:
    merged_signatures = merged_signatures_extend
    
# Add number of regions
merged_signatures = {
    x + '_(' + str(len(merged_signatures[x])) + 'r)': merged_signatures[x] for x in merged_signatures.keys()}

merged_signatures.keys() # Cistromes with no extension contain regions linked to directly annotated motifs, while ‘_extended’ cistromes can contain regions linked to motifs annotated by similarity or orthology.
with open(save_dir + "/cistromes.pkl", "wb") as tf:
    pickle.dump(merged_signatures, tf)
with open(save_dir + "/cistromes.pkl", "rb") as tf:
    cistromes = pickle.load(tf)
    
# filter tf with no-expression
filtered_cistromes = dict()
for key in cistromes.keys():
    tf_name = key.split("_")[0]
    if tf_name in adata_r.var_names:
        filtered_cistromes[key] = cistromes[key]
filtered_cistromes.keys()

with open(save_dir + "/filtered_cistromes.pkl", "wb") as tf:
    pickle.dump(filtered_cistromes, tf)
with open(save_dir + "/filtered_cistromes.pkl", "rb") as tf:
    _ = pickle.load(tf)

"""


"""
cistarget_dict = run_cistarget(ctx_db = '/home/nomura/Proj/mmvelo/pycistarget/mm10_screen_v10_clust.regions_vs_motifs.rankings.feather',
                                                      region_sets = region_sets,
                                                      specie = 'mus_musculus',
                                                      auc_threshold = 0.005,
                                                      nes_threshold = 3.0,
                                                      rank_threshold = 0.05,
                                                      annotation = ['Direct_annot', 'Orthology_annot'],
                                                      annotation_version = 'v10nr_clust',
                                                      path_to_motif_annotations = '/home/nomura/Proj/mmvelo/pycistarget/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl',
                                                      n_cpu = 4,
                                                      #_temp_dir='/scratch/leuven/313/vsc31305/ray_spill'
                                                      )

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
with open(dir_path + '/cisTarget_dict.pkl', 'wb') as f:
  pickle.dump(cistarget_dict, f)

infile = open(dir_path + '/cisTarget_dict.pkl', 'rb')
cistarget_dict = pickle.load(infile)
infile.close()

#cistarget_results(cistarget_dict, name='0')
for clst in set(adata_peak.obs["leiden"]):
    out_file = dir_path + f'/cluster_{clst}_motif_enricment.html'
    cistarget_dict[clst].motif_enrichment.to_html(open(out_file, 'w'), escape=False, col_space=80)

# note;  heatmap is depicted in cluster_order = [4, 6, 5, 0, 1, 2, 7, 3, 8]
# TF motifs with corresponding tf expression?



# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
pseudotime = pd.read_csv(dir_path+"/pseudotime.tsv", sep="\t", header=None)[0].to_numpy()
adata_r.obs["pseudotime"] = pseudotime

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
for clst in set(adata_peak.obs["leiden"]):

    tf_motifs = cistarget_dict[clst].motif_enrichment
    #tf_lists = list(cistarget_dict[clst].motif_enrichment["Direct_annot"][:10]) + \
    #        list(cistarget_dict[clst].motif_enrichment["Orthology_annot"][:10])
    tf_lists = list(cistarget_dict[clst].motif_enrichment["Direct_annot"]) + \
            list(cistarget_dict[clst].motif_enrichment["Orthology_annot"])

    tf_list = []
    for i in range(len(tf_lists)):
        if isinstance(tf_lists[i], float):
            continue
        tf_lists[i] = tf_lists[i].split(", ")
        tf_list.append(tf_lists[i])
    tf_list = sum(tf_list, [])

    exists_tf_list = []
    for tf in tf_list:
        if tf in adata_r.var_names:
            exists_tf_list.append(tf)
    exists_tf_list = list(set(exists_tf_list))
    
    with open(dir_path + "/exist_tf_motifcluster_{}.txt".format(clst), "w") as output:
        output.write(str(exists_tf_list))
    
    

adata_r
lowess = sm.nonparametric.lowess
frac = 300 / adata_r.n_obs # use 10 neighbor cells for lowess regression
p_time = np.linspace(0, 1, adata_r.n_obs)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
for clst in set(adata_peak.obs["leiden"]):
    adata_peak_clst = adata_peak[adata_peak.obs["leiden"] == clst, :]
    peak_clst_dadt = adata_peak_clst.X.toarray().mean(0)

    p_time = np.linspace(0, 1, peak_clst_dadt.shape[0])
    smooth_peak_dadt = lowess(peak_clst_dadt, p_time, frac=frac, return_sorted=False)
    
    norm_peak_clst_dadt = peak_clst_dadt / np.max(np.abs(smooth_peak_dadt))
    norm_smooth_peak_dadt = smooth_peak_dadt / np.max(np.abs(smooth_peak_dadt))
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(p_time, norm_smooth_peak_dadt, color="blue", linewidth=3)
    ax.scatter(p_time, norm_peak_clst_dadt, s=0.1, color="blue")
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('pseudotime')
    ax.set_ylabel('mean velocity')
    ax.set_title(f'Mean velocity in cluster {clst}')
    fig.tight_layout()
    plt.savefig(dir_path + f"/clst_{clst}_mean_vel.png", bbox_inches='tight', dpi=300)
    plt.close()

    tf_motifs = cistarget_dict[clst].motif_enrichment

    tf_lists = list(cistarget_dict[clst].motif_enrichment["Direct_annot"][:10]) + \
            list(cistarget_dict[clst].motif_enrichment["Orthology_annot"][:10])

    tf_list = []
    for i in range(len(tf_lists)):
        if isinstance(tf_lists[i], float):
            continue
        tf_lists[i] = tf_lists[i].split(", ")
        tf_list.append(tf_lists[i])
    tf_list = sum(tf_list, [])

    exists_tf_list = []
    for tf in tf_list:
        if tf in adata_r.var_names:
            exists_tf_list.append(tf)


    dir_path = dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
    file_path = dir_path + f"/tf_in_cluster_{clst}"
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    for tf in exists_tf_list:
        argsort = np.argsort(adata_r.obs["pseudotime"])
        p_time = np.linspace(0, 1, adata_r.n_obs)
        tf_s = adata_r[argsort, tf].layers["s_raw"].toarray().reshape(-1)
        tf_s = tf_s / np.std(tf_s)
        smooth_tf_s = lowess(tf_s, p_time, frac=frac, return_sorted=False)

        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(p_time, tf_s, s=0.1, color = "red")
        ax.plot(p_time, smooth_tf_s, color="red", linewidth=3)

        ax.scatter(p_time, peak_clst_dadt, s=0.1, color="blue")
        ax.plot(p_time, smooth_peak_dadt, color="blue", linewidth=3)
        ax.set_xlabel('pseudotime')
        ax.set_ylabel('mean expr/vel')
        ax.set_title(f'{tf}  mean expr vs {clst} mean peak vel')
        fig.tight_layout()
        plt.savefig(file_path + f"/{tf}_mean_expr.png", bbox_inches='tight')
        plt.close()
        

        fig, ax = plt.subplots(figsize=(5, 5))
        norm_tf_s = tf_s / max(smooth_tf_s)
        norm_smooth_tf_s = smooth_tf_s / max(smooth_tf_s)
        norm_peak_clst_dadt = peak_clst_dadt / np.max(np.abs(smooth_peak_dadt))
        norm_smooth_peak_dadt = smooth_peak_dadt / np.max(np.abs(smooth_peak_dadt))

        ax.scatter(p_time, norm_tf_s, s=0.1, color="red")
        ax.plot(p_time, norm_smooth_tf_s, color="red", linewidth=3)
        ax.scatter(p_time, norm_peak_clst_dadt, s=0.1, color="blue")
        ax.plot(p_time, norm_smooth_peak_dadt, color="blue", linewidth=3)
        ax.set_ylim(-1.5, 1.5)

        ax.set_xlabel('pseudotime')
        ax.set_ylabel('norm mean expr/vel')
        ax.set_title(f'{tf}  mean expr vs {clst} mean peak vel')
        fig.tight_layout()
        plt.savefig(file_path + f"/{tf}_mean_expr_norm.png", bbox_inches='tight')
        plt.close()

        # for fig
        
        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(p_time, norm_tf_s, s=0.1, color="red")
        ax.plot(p_time, norm_smooth_tf_s, color="red", linewidth=3)
        ax.scatter(p_time, norm_peak_clst_dadt, s=0.1, color="blue")
        ax.plot(p_time, norm_smooth_peak_dadt, color="blue", linewidth=3)
        ax.set_ylim(-1.5, 1.5)

        ax.set_xlabel('pseudotime')
        ax.set_ylabel('norm mean expr/vel')
        plt.gca().axis("off")
        fig.tight_layout()
        plt.savefig(file_path + f"/{tf}_mean_expr_norm_blank.png", bbox_inches='tight', dpi=300)
        plt.close()
        
"""