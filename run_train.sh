# prepare data
mkdir data
unzip -d data /kqi/input/training/22469060/nagasaki-cluster.zip
mkdir add_data
unzip -d add_data /kqi/parent/22022628/79WSIs_patch.zip
mkdir add_data2
unzip -d add_data2 /kqi/input/training/22469090/NoCarcinomaPatch.zip

# run cross validation for cluster1
for i in {0..4} ; do
    python train.py \
data_dir=data/cluster1 \
add_data_dir=add_data/79WSIs_patch \
add_data_dir2=add_data2/NoCarcinomaPatch \
output_dir=/kqi/output/fold${i} fold=${i}
done

python validation.py data_dir=data/cluster1 model_dir=/kqi/output output_dir=/kqi/output

# run cross validation for cluster2
for i in {0..4} ; do
    python train.py \
data_dir=data/cluster2 \
add_data_dir=add_data/79WSIs_patch \
add_data_dir2=add_data2/NoCarcinomaPatch \
output_dir=/kqi/output/fold${i} fold=${i}
done

python validation.py data_dir=data/cluster2 model_dir=/kqi/output output_dir=/kqi/output


# inference
python inference.py /kqi/input /kqi/output Test_Inference
