echo "\nlive"
# live
python calc_scores.py -d live -f dinov1_l2
python calc_scores.py -d live -f dinov1_cos


python calc_scores.py -d live -f openclip_convnext_base_cos
python calc_scores.py -d live -f dinov2_cos
python calc_scores.py -d live -f embed_clip_vitb32_cos


# tid2013
python calc_scores.py -d tid2013 -f dinov1_l2
python calc_scores.py -d tid2013 -f dinov1_cos


python calc_scores.py -d tid2013 -f openclip_convnext_base_cos
python calc_scores.py -d tid2013 -f dinov2_cos
python calc_scores.py -d tid2013 -f embed_clip_vitb32_cos


# pipal
python calc_scores.py -d pipal -f dinov1_l2
python calc_scores.py -d pipal -f dinov1_cos


python calc_scores.py -d pipal -f openclip_convnext_base_cos
python calc_scores.py -d pipal -f dinov2_cos
python calc_scores.py -d pipal -f embed_clip_vitb32_cos


##### rob

echo "\nlive tra"
## live
python calc_scores.py -d live -f dinov1_l2_tra-1
python calc_scores.py -d live -f dinov1_cos_tra-1

python calc_scores.py -d live -f openclip_convnext_base_cos_tra-1
python calc_scores.py -d live -f dinov2_cos_tra-1
python calc_scores.py -d live -f embed_clip_vitb32_cos_tra-1
echo "\nlive sca"
#sca
python calc_scores.py -d live -f dinov1_l2_sca-1
python calc_scores.py -d live -f dinov1_cos_sca-1

python calc_scores.py -d live -f openclip_convnext_base_cos_sca-1
python calc_scores.py -d live -f dinov2_cos_sca-1
python calc_scores.py -d live -f embed_clip_vitb32_cos_sca-1
echo "\nlive rot"
#rot
python calc_scores.py -d live -f dinov1_l2_rot-1
python calc_scores.py -d live -f dinov1_cos_rot-1

python calc_scores.py -d live -f openclip_convnext_base_cos_rot-1
python calc_scores.py -d live -f dinov2_cos_rot-1
python calc_scores.py -d live -f embed_clip_vitb32_cos_rot-1

echo "\ntid2013 tra"
## tid2013
python calc_scores.py -d tid2013 -f dinov1_l2_tra-1
python calc_scores.py -d tid2013 -f dinov1_cos_tra-1

python calc_scores.py -d tid2013 -f openclip_convnext_base_cos_tra-1
python calc_scores.py -d tid2013 -f dinov2_cos_tra-1
python calc_scores.py -d tid2013 -f embed_clip_vitb32_cos_tra-1

echo "\ntid2013 sca"
#sca
python calc_scores.py -d tid2013 -f dinov1_l2_sca-1
python calc_scores.py -d tid2013 -f dinov1_cos_sca-1

python calc_scores.py -d tid2013 -f openclip_convnext_base_cos_sca-1
python calc_scores.py -d tid2013 -f dinov2_cos_sca-1
python calc_scores.py -d tid2013 -f embed_clip_vitb32_cos_sca-1

echo "\ntid2013 rot"
#rot
python calc_scores.py -d tid2013 -f dinov1_l2_rot-1
python calc_scores.py -d tid2013 -f dinov1_cos_rot-1

python calc_scores.py -d tid2013 -f openclip_convnext_base_cos_rot-1
python calc_scores.py -d tid2013 -f dinov2_cos_rot-1
python calc_scores.py -d tid2013 -f embed_clip_vitb32_cos_rot-1

echo "\npipal tra"
## pipal
python calc_scores.py -d pipal -f dinov1_l2_tra-1
python calc_scores.py -d pipal -f dinov1_cos_tra-1

python calc_scores.py -d pipal -f openclip_convnext_base_cos_tra-1
python calc_scores.py -d pipal -f dinov2_cos_tra-1
python calc_scores.py -d pipal -f embed_clip_vitb32_cos_tra-1

echo "\npipal sca"
#sca
python calc_scores.py -d pipal -f dinov1_l2_sca-1
python calc_scores.py -d pipal -f dinov1_cos_sca-1

python calc_scores.py -d pipal -f openclip_convnext_base_cos_sca-1
python calc_scores.py -d pipal -f dinov2_cos_sca-1
python calc_scores.py -d pipal -f embed_clip_vitb32_cos_sca-1

echo "\npipal rot"
#rot
python calc_scores.py -d pipal -f dinov1_l2_rot-1
python calc_scores.py -d pipal -f dinov1_cos_rot-1

python calc_scores.py -d pipal -f openclip_convnext_base_cos_rot-1
python calc_scores.py -d pipal -f dinov2_cos_rot-1
python calc_scores.py -d pipal -f embed_clip_vitb32_cos_rot-1


##### others

# live
echo "live"
python calc_scores.py -d live -f dinov1_l2
python calc_scores.py -d live -f dinov1_cos
python calc_scores.py -d live -f dinov1_skld
python calc_scores.py -d live -f dinov1_wsd
python calc_scores.py -d live -f dinov1_jsd
python calc_scores.py -d live -f dinov1_swd
python calc_scores.py -d live -f openclip_convnext_base_cos
python calc_scores.py -d live -f dinov2_cos
python calc_scores.py -d live -f embed_clip_vitb32_cos
python calc_scores.py -d live -f embed_dinov1_cos
python calc_scores.py -d live -f embed_dinov2_cos
python calc_scores.py -d live -f embed_openclip_convnext_base_cos
python calc_scores.py -d live -f clip_rn50_cos
python calc_scores.py -d live -f embed_clip_rn50_cos

# tid2013
echo "tid2013"
python calc_scores.py -d tid2013 -f dinov1_l2
python calc_scores.py -d tid2013 -f dinov1_cos
python calc_scores.py -d tid2013 -f dinov1_skld
python calc_scores.py -d tid2013 -f dinov1_wsd
python calc_scores.py -d tid2013 -f dinov1_jsd
python calc_scores.py -d tid2013 -f dinov1_swd
python calc_scores.py -d tid2013 -f openclip_convnext_base_cos
python calc_scores.py -d tid2013 -f dinov2_cos
python calc_scores.py -d tid2013 -f embed_clip_vitb32_cos
python calc_scores.py -d tid2013 -f embed_dinov1_cos
python calc_scores.py -d tid2013 -f embed_dinov2_cos
python calc_scores.py -d tid2013 -f embed_openclip_convnext_base_cos
python calc_scores.py -d tid2013 -f clip_rn50_cos
python calc_scores.py -d tid2013 -f embed_clip_rn50_cos

# pipal
echo "pipal"
python calc_scores.py -d pipal -f dinov1_l2
python calc_scores.py -d pipal -f dinov1_cos
python calc_scores.py -d pipal -f dinov1_skld
python calc_scores.py -d pipal -f dinov1_wsd
python calc_scores.py -d pipal -f dinov1_jsd
python calc_scores.py -d pipal -f dinov1_swd
python calc_scores.py -d pipal -f openclip_convnext_base_cos
python calc_scores.py -d pipal -f dinov2_cos
python calc_scores.py -d pipal -f embed_clip_vitb32_cos
python calc_scores.py -d pipal -f embed_dinov1_cos
python calc_scores.py -d pipal -f embed_dinov2_cos
python calc_scores.py -d pipal -f embed_openclip_convnext_base_cos
python calc_scores.py -d pipal -f clip_rn50_cos
python calc_scores.py -d pipal -f embed_clip_rn50_cos
python calc_scores.py -d pipal -f clip_rn50_l2
python calc_scores.py -d pipal -f openclip_convnext_base_l2
python calc_scores.py -d pipal -f dinov2_l2


# ##### rob


## live
echo "live"
python calc_scores.py -d live -f dinov1_l2_tra-1
python calc_scores.py -d live -f dinov1_cos_tra-1
python calc_scores.py -d live -f dinov1_skld_tra-1
python calc_scores.py -d live -f dinov1_wsd_tra-1
python calc_scores.py -d live -f dinov1_jsd_tra-1
python calc_scores.py -d live -f dinov1_swd_tra-1
python calc_scores.py -d live -f openclip_convnext_base_cos_tra-1
python calc_scores.py -d live -f dinov2_cos_tra-1
python calc_scores.py -d live -f embed_clip_vitb32_cos_tra-1
python calc_scores.py -d live -f embed_dinov1_cos_tra-1
python calc_scores.py -d live -f embed_dinov2_cos_tra-1
python calc_scores.py -d live -f embed_openclip_convnext_base_cos_tra-1
python calc_scores.py -d live -f clip_rn50_cos_tra-1
python calc_scores.py -d live -f embed_clip_rn50_cos_tra-1


#sca
python calc_scores.py -d live -f dinov1_l2_sca-1
python calc_scores.py -d live -f dinov1_cos_sca-1
python calc_scores.py -d live -f dinov1_skld_sca-1
python calc_scores.py -d live -f dinov1_wsd_sca-1
python calc_scores.py -d live -f dinov1_jsd_sca-1
python calc_scores.py -d live -f dinov1_swd_sca-1
python calc_scores.py -d live -f openclip_convnext_base_cos_sca-1
python calc_scores.py -d live -f dinov2_cos_sca-1
python calc_scores.py -d live -f embed_clip_vitb32_cos_sca-1
python calc_scores.py -d live -f embed_dinov1_cos_sca-1
python calc_scores.py -d live -f embed_dinov2_cos_sca-1
python calc_scores.py -d live -f embed_openclip_convnext_base_cos_sca-1
python calc_scores.py -d live -f clip_rn50_cos_sca-1
python calc_scores.py -d live -f embed_clip_rn50_cos_sca-1

#rot
python calc_scores.py -d live -f dinov1_l2_rot-1
python calc_scores.py -d live -f dinov1_cos_rot-1
python calc_scores.py -d live -f dinov1_skld_rot-1
python calc_scores.py -d live -f dinov1_wsd_rot-1
python calc_scores.py -d live -f dinov1_jsd_rot-1
python calc_scores.py -d live -f dinov1_swd_rot-1
python calc_scores.py -d live -f openclip_convnext_base_cos_rot-1
python calc_scores.py -d live -f dinov2_cos_rot-1
python calc_scores.py -d live -f embed_clip_vitb32_cos_rot-1
python calc_scores.py -d live -f embed_dinov1_cos_rot-1
python calc_scores.py -d live -f embed_dinov2_cos_rot-1
python calc_scores.py -d live -f embed_openclip_convnext_base_cos_rot-1
python calc_scores.py -d live -f clip_rn50_cos_rot-1
python calc_scores.py -d live -f embed_clip_rn50_cos_rot-1



## tid2013
echo "tid2013"
python calc_scores.py -d tid2013 -f dinov1_l2_tra-1
python calc_scores.py -d tid2013 -f dinov1_cos_tra-1
python calc_scores.py -d tid2013 -f dinov1_skld_tra-1
python calc_scores.py -d tid2013 -f dinov1_wsd_tra-1
python calc_scores.py -d tid2013 -f dinov1_jsd_tra-1
python calc_scores.py -d tid2013 -f dinov1_swd_tra-1
python calc_scores.py -d tid2013 -f openclip_convnext_base_cos_tra-1
python calc_scores.py -d tid2013 -f dinov2_cos_tra-1
python calc_scores.py -d tid2013 -f embed_clip_vitb32_cos_tra-1
python calc_scores.py -d tid2013 -f embed_dinov1_cos_tra-1
python calc_scores.py -d tid2013 -f embed_dinov2_cos_tra-1
python calc_scores.py -d tid2013 -f embed_openclip_convnext_base_cos_tra-1
python calc_scores.py -d tid2013 -f clip_rn50_cos_tra-1
python calc_scores.py -d tid2013 -f embed_clip_rn50_cos_tra-1

#sca
python calc_scores.py -d tid2013 -f dinov1_l2_sca-1
python calc_scores.py -d tid2013 -f dinov1_cos_sca-1
python calc_scores.py -d tid2013 -f dinov1_skld_sca-1
python calc_scores.py -d tid2013 -f dinov1_wsd_sca-1
python calc_scores.py -d tid2013 -f dinov1_jsd_sca-1
python calc_scores.py -d tid2013 -f dinov1_swd_sca-1
python calc_scores.py -d tid2013 -f openclip_convnext_base_cos_sca-1
python calc_scores.py -d tid2013 -f dinov2_cos_sca-1
python calc_scores.py -d tid2013 -f embed_clip_vitb32_cos_sca-1
python calc_scores.py -d tid2013 -f embed_dinov1_cos_sca-1
python calc_scores.py -d tid2013 -f embed_dinov2_cos_sca-1
python calc_scores.py -d tid2013 -f embed_openclip_convnext_base_cos_sca-1
python calc_scores.py -d tid2013 -f clip_rn50_cos_sca-1
python calc_scores.py -d tid2013 -f embed_clip_rn50_cos_sca-1

#rot
python calc_scores.py -d tid2013 -f dinov1_l2_rot-1
python calc_scores.py -d tid2013 -f dinov1_cos_rot-1
python calc_scores.py -d tid2013 -f dinov1_skld_rot-1
python calc_scores.py -d tid2013 -f dinov1_wsd_rot-1
python calc_scores.py -d tid2013 -f dinov1_jsd_rot-1
python calc_scores.py -d tid2013 -f dinov1_swd_rot-1
python calc_scores.py -d tid2013 -f openclip_convnext_base_cos_rot-1
python calc_scores.py -d tid2013 -f dinov2_cos_rot-1
python calc_scores.py -d tid2013 -f embed_clip_vitb32_cos_rot-1
python calc_scores.py -d tid2013 -f embed_dinov1_cos_rot-1
python calc_scores.py -d tid2013 -f embed_dinov2_cos_rot-1
python calc_scores.py -d tid2013 -f embed_openclip_convnext_base_cos_rot-1
python calc_scores.py -d tid2013 -f clip_rn50_cos_rot-1
python calc_scores.py -d tid2013 -f embed_clip_rn50_cos_rot-1



## pipal
echo "pipal"
python calc_scores.py -d pipal -f dinov1_l2_tra-1
python calc_scores.py -d pipal -f dinov1_cos_tra-1
python calc_scores.py -d pipal -f dinov1_skld_tra-1
python calc_scores.py -d pipal -f dinov1_wsd_tra-1
python calc_scores.py -d pipal -f dinov1_jsd_tra-1
python calc_scores.py -d pipal -f dinov1_swd_tra-1
python calc_scores.py -d pipal -f openclip_convnext_base_cos_tra-1
python calc_scores.py -d pipal -f dinov2_cos_tra-1
python calc_scores.py -d pipal -f embed_clip_vitb32_cos_tra-1
python calc_scores.py -d pipal -f embed_dinov1_cos_tra-1
python calc_scores.py -d pipal -f embed_dinov2_cos_tra-1
python calc_scores.py -d pipal -f embed_openclip_convnext_base_cos_tra-1
python calc_scores.py -d pipal -f clip_rn50_cos_tra-1
python calc_scores.py -d pipal -f embed_clip_rn50_cos_tra-1
python calc_scores.py -d pipal -f clip_rn50_l2_tra-1
python calc_scores.py -d pipal -f openclip_convnext_base_l2_tra-1
python calc_scores.py -d pipal -f dinov2_l2_tra-1

#sca
python calc_scores.py -d pipal -f dinov1_l2_sca-1
python calc_scores.py -d pipal -f dinov1_cos_sca-1
python calc_scores.py -d pipal -f dinov1_skld_sca-1
python calc_scores.py -d pipal -f dinov1_wsd_sca-1
python calc_scores.py -d pipal -f dinov1_jsd_sca-1
python calc_scores.py -d pipal -f dinov1_swd_sca-1
python calc_scores.py -d pipal -f openclip_convnext_base_cos_sca-1
python calc_scores.py -d pipal -f dinov2_cos_sca-1
python calc_scores.py -d pipal -f embed_clip_vitb32_cos_sca-1
python calc_scores.py -d pipal -f embed_dinov1_cos_sca-1
python calc_scores.py -d pipal -f embed_dinov2_cos_sca-1
python calc_scores.py -d pipal -f embed_openclip_convnext_base_cos_sca-1
python calc_scores.py -d pipal -f clip_rn50_cos_sca-1
python calc_scores.py -d pipal -f embed_clip_rn50_cos_sca-1
python calc_scores.py -d pipal -f clip_rn50_l2_sca-1
python calc_scores.py -d pipal -f openclip_convnext_base_l2_sca-1
python calc_scores.py -d pipal -f dinov2_l2_sca-1

# #rot
python calc_scores.py -d pipal -f dinov1_l2_rot-1
python calc_scores.py -d pipal -f dinov1_cos_rot-1
python calc_scores.py -d pipal -f dinov1_skld_rot-1
python calc_scores.py -d pipal -f dinov1_wsd_rot-1
python calc_scores.py -d pipal -f dinov1_jsd_rot-1
python calc_scores.py -d pipal -f dinov1_swd_rot-1
python calc_scores.py -d pipal -f openclip_convnext_base_cos_rot-1
python calc_scores.py -d pipal -f dinov2_cos_rot-1
python calc_scores.py -d pipal -f embed_clip_vitb32_cos_rot-1
python calc_scores.py -d pipal -f embed_dinov1_cos_rot-1
python calc_scores.py -d pipal -f embed_dinov2_cos_rot-1
python calc_scores.py -d pipal -f embed_openclip_convnext_base_cos_rot-1
python calc_scores.py -d pipal -f clip_rn50_cos_rot-1
python calc_scores.py -d pipal -f embed_clip_rn50_cos_rot-1
python calc_scores.py -d pipal -f clip_rn50_l2_rot-1
python calc_scores.py -d pipal -f openclip_convnext_base_l2_rot-1
python calc_scores.py -d pipal -f dinov2_l2_rot-1