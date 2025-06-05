semantic_seg_dir=segmentations/cell-classes

pp_dir=post-processing

pp0=$pp_dir/fix_outside.py
pp0_indir=$semantic_seg_dir
pp0_outdir=$semantic_seg_dir/00-fixed-outside
pp0_args="--sigma 1 --epidermis-id 4 --outside-id 2"

mkdir -p $pp0_outdir

col0_w3="008_col0_w3_p0_l6b_6_zoomed-0.25.nii.gz 009_col0_w3_p0_l6m_zoomed-0.25.nii.gz 010_col0_w3_p0_l6t_zoomed-0.25.nii.gz 011_col0_w3_p0_l6t_2_zoomed-0.25.nii.gz 014_col0_w3_p0_l8b_3_zoomed-0.25.nii.gz 015_col0_w3_p0_l8m_zoomed-0.25.nii.gz 016_col0_w3_p0_l8t_zoomed-0.25.nii.gz 017_col0_w3_p1_l6b_zoomed-0.25.nii.gz 018_col0_w3_p1_l6m_zoomed-0.25.nii.gz 019_col0_w3_p1_l6t_zoomed-0.25.nii.gz 021_col0_w3_p1_l7b_2_zoomed-0.25.nii.gz 022_col0_w3_p1_l7m_zoomed-0.25.nii.gz 023_col0_w3_p1_l7t_zoomed-0.25.nii.gz"

col0_w6="149_Col0_w6_p1_l6b_zoomed-0.25.nii.gz 151_Col0_w6_p1_l6m_2_zoomed-0.25.nii.gz 152_Col0_w6_p1_l6t_zoomed-0.25.nii.gz 153_Col0_w6_p2_l7b_zoomed-0.25.nii.gz 155_Col0_w6_p1_l7m_zoomed-0.25.nii.gz 156_Col0_w6_p1_l7t_zoomed-0.25.nii.gz 157_Col0_w6_p2_l6b_zoomed-0.25.nii.gz 158_Col0_w6_p2_l6m_zoomed-0.25.nii.gz 159_Col0_w6_p2_l6t_zoomed-0.25.nii.gz 160_Col0_w6_p2_l7b_zoomed-0.25.nii.gz 161_Col0_w6_p2_l7m_zoomed-0.25.nii.gz 162_Col0_w6_p2_l7t_zoomed-0.25.nii.gz"

ric_w3="024_RIC_w3_p4_l6b_zoomed-0.25.nii.gz 025_RIC_w3_p4_l6m_zoomed-0.25.nii.gz 026_RIC_w3_p4_l6t_zoomed-0.25.nii.gz 027_RIC_w3_p4_l7b_zoomed-0.25.nii.gz 028_RIC_w3_p4_l7m_zoomed-0.25.nii.gz 029_RIC_w3_p4_l7t_zoomed-0.25.nii.gz 030_RIC_w3_p2_l6b_zoomed-0.25.nii.gz 031_RIC_w3_p2_l6m_zoomed-0.25.nii.gz 032_RIC_w3_p2_l6t_zoomed-0.25.nii.gz"

ric_w6="136_RIC_w6_p2_l7b_zoomed-0.25.nii.gz 137_RIC_w6_p2_l7m_zoomed-0.25.nii.gz 138_RIC_w6_p2_l7t_zoomed-0.25.nii.gz 139_RIC_w6_p2_l8b_zoomed-0.25.nii.gz 140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz 141_RIC_w6_p2_l8t_zoomed-0.25.nii.gz 143_RIC_w6_p1_l6b_2_zoomed-0.25.nii.gz 144_RIC_w6_p1_l6m_zoomed-0.25.nii.gz 145_RIC_w6_p1_l6t_zoomed-0.25.nii.gz 146_RIC_w6_p1_l7b_zoomed-0.25.nii.gz 147_RIC_w6_p1_l7m_zoomed-0.25.nii.gz 148_RIC_w6_p1_l7t_zoomed-0.25.nii.gz"

rop_w3="034_ROP_w3_p2_l6b_2_zoomed-0.25.nii.gz 035_ROP_w3_p2_l6m_zoomed-0.25.nii.gz 036_ROP_w3_p2_l6t_zoomed-0.25.nii.gz 037_ROP_w3_p2_l7b_zoomed-0.25.nii.gz 038_ROP_w3_p2_l7m_zoomed-0.25.nii.gz 039_ROP_w3_p2_l7t_zoomed-0.25.nii.gz 040_ROP_w3_p1_l6b_zoomed-0.25.nii.gz 041_ROP_w3_p1_l6m_zoomed-0.25.nii.gz 042_ROP_w3_p1_l6t_zoomed-0.25.nii.gz 043_ROP_w3_p1_l7b_zoomed-0.25.nii.gz 044_ROP_w3_p1_l7m_zoomed-0.25.nii.gz 045_ROP_w3_p1_l7t_zoomed-0.25.nii.gz"

rop_w6="124_ROP_w6_p1_l6b_zoomed-0.25.nii.gz 125_ROP_w6_p1_l6m_zoomed-0.25.nii.gz 126_ROP_w6_p1_l6t_zoomed-0.25.nii.gz 127_ROP_w6_p1_l7b_zoomed-0.25.nii.gz 128_ROP_w6_p1_l7m_zoomed-0.25.nii.gz 129_ROP_w6_p1_l7t_zoomed-0.25.nii.gz 130_ROP_w6_p1_l8b_zoomed-0.25.nii.gz 131_ROP_w6_p1_l8m_zoomed-0.25.nii.gz 132_ROP_w6_p1_l8t_zoomed-0.25.nii.gz 133_ROP_w6_p2_l7b_zoomed-0.25.nii.gz 134_ROP_w6_p2_l7m_zoomed-0.25.nii.gz 135_ROP_w6_p2_l7t_zoomed-0.25.nii.gz 163_ROP_w6_p2_l7b_zoomed-0.25.nii.gz 164_ROP_w6_p2_l7m_zoomed-0.25.nii.gz 165_ROP_w6_p2_l7t_zoomed-0.25.nii.gz"

images="$ric_w3 $ric_w6 $rop_w3 $rop_w6"

for image in $images;
do
    python3 $pp0 $pp0_indir/$image $pp0_outdir/$image $pp0_args
done
