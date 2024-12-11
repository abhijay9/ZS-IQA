echo "\nlive"
python calc_scores.py -d live -f deep_kld_vgg
python calc_scores.py -d live -f deep_jsd_vgg
python calc_scores.py -d live -f deep_wsd_vgg
python calc_scores.py -d live -f deep_kld_eff
python calc_scores.py -d live -f deep_jsd_eff
python calc_scores.py -d live -f deep_wsd_eff

python calc_scores.py -d live -f clip_vitb32_l2
python calc_scores.py -d live -f clip_vitb32_cos
python calc_scores.py -d live -f clip_vitb32_swd
python calc_scores.py -d live -f clip_vitb32_skld
python calc_scores.py -d live -f clip_vitb32_wsd
python calc_scores.py -d live -f clip_vitb32_jsd

python calc_scores.py -d live -f psnr
python calc_scores.py -d live -f msssim
python calc_scores.py -d live -f fsim
python calc_scores.py -d live -f dists
python calc_scores.py -d live -f lpips
python calc_scores.py -d live -f lpips_vgg
python calc_scores.py -d live -f stlpips_alex
python calc_scores.py -d live -f stlpips_vgg
python calc_scores.py -d live -f lpips_vgg_kadid
python calc_scores.py -d live -f stlpips_vgg_kadid



echo "\ntid2013"
python calc_scores.py -d tid2013 -f deep_kld_vgg
python calc_scores.py -d tid2013 -f deep_jsd_vgg
python calc_scores.py -d tid2013 -f deep_wsd_vgg
python calc_scores.py -d tid2013 -f deep_kld_eff
python calc_scores.py -d tid2013 -f deep_jsd_eff
python calc_scores.py -d tid2013 -f deep_wsd_eff

python calc_scores.py -d tid2013 -f clip_vitb32_l2
python calc_scores.py -d tid2013 -f clip_vitb32_cos
python calc_scores.py -d tid2013 -f clip_vitb32_swd
python calc_scores.py -d tid2013 -f clip_vitb32_skld
python calc_scores.py -d tid2013 -f clip_vitb32_wsd
python calc_scores.py -d tid2013 -f clip_vitb32_jsd

python calc_scores.py -d tid2013 -f psnr
python calc_scores.py -d tid2013 -f msssim
python calc_scores.py -d tid2013 -f fsim
python calc_scores.py -d tid2013 -f dists
python calc_scores.py -d tid2013 -f lpips
python calc_scores.py -d tid2013 -f lpips_vgg
python calc_scores.py -d tid2013 -f stlpips_alex
python calc_scores.py -d tid2013 -f stlpips_vgg
python calc_scores.py -d tid2013 -f lpips_vgg_kadid
python calc_scores.py -d tid2013 -f stlpips_vgg_kadid



echo "\npipal"
python calc_scores.py -d pipal -f deep_kld_vgg
python calc_scores.py -d pipal -f deep_jsd_vgg
python calc_scores.py -d pipal -f deep_wsd_vgg
python calc_scores.py -d pipal -f deep_kld_eff
python calc_scores.py -d pipal -f deep_jsd_eff
python calc_scores.py -d pipal -f deep_wsd_eff

python calc_scores.py -d pipal -f clip_vitb32_l2
python calc_scores.py -d pipal -f clip_vitb32_cos
python calc_scores.py -d pipal -f clip_vitb32_swd
python calc_scores.py -d pipal -f clip_vitb32_skld
python calc_scores.py -d pipal -f clip_vitb32_wsd
python calc_scores.py -d pipal -f clip_vitb32_jsd

python calc_scores.py -d pipal -f psnr
python calc_scores.py -d pipal -f msssim
python calc_scores.py -d pipal -f fsim
python calc_scores.py -d pipal -f dists
python calc_scores.py -d pipal -f lpips
python calc_scores.py -d pipal -f lpips_vgg
python calc_scores.py -d pipal -f stlpips_alex
python calc_scores.py -d pipal -f stlpips_vgg
python calc_scores.py -d pipal -f lpips_vgg_kadid
python calc_scores.py -d pipal -f stlpips_vgg_kadid




echo "live"
# live
python calc_scores.py -d live -f imagenet_vit_cos
python calc_scores.py -d live -f imagenet_vit_l2
python calc_scores.py -d live -f openclip_vitb32_laion_tecoa4_cos
python calc_scores.py -d live -f embed_openclip_vitb32_laion_tecoa4_cos
python calc_scores.py -d live -f openclip_convnext_laion_tecoa4_cos
python calc_scores.py -d live -f embed_openclip_convnext_laion_tecoa4_cos

echo "tid2013"
# tid2013
python calc_scores.py -d tid2013 -f imagenet_vit_cos
python calc_scores.py -d tid2013 -f imagenet_vit_l2
python calc_scores.py -d tid2013 -f openclip_vitb32_laion_tecoa4_cos
python calc_scores.py -d tid2013 -f embed_openclip_vitb32_laion_tecoa4_cos
python calc_scores.py -d tid2013 -f openclip_convnext_laion_tecoa4_cos
python calc_scores.py -d tid2013 -f embed_openclip_convnext_laion_tecoa4_cos

echo "pipal"
# pipal
python calc_scores.py -d pipal -f imagenet_vit_cos
python calc_scores.py -d pipal -f imagenet_vit_l2
python calc_scores.py -d pipal -f openclip_vitb32_laion_tecoa4_cos
python calc_scores.py -d pipal -f embed_openclip_vitb32_laion_tecoa4_cos
python calc_scores.py -d pipal -f openclip_convnext_laion_tecoa4_cos
python calc_scores.py -d pipal -f embed_openclip_convnext_laion_tecoa4_cos

##### rob
echo "live"
# live
python calc_scores.py -d live -f imagenet_vit_cos_tra-1
python calc_scores.py -d live -f imagenet_vit_l2_tra-1
python calc_scores.py -d live -f openclip_vitb32_laion_tecoa4_cos_tra-1
python calc_scores.py -d live -f embed_openclip_vitb32_laion_tecoa4_cos_tra-1
python calc_scores.py -d live -f openclip_convnext_laion_tecoa4_cos_tra-1
python calc_scores.py -d live -f embed_openclip_convnext_laion_tecoa4_cos_tra-1

echo "tid2013"
# tid2013
python calc_scores.py -d tid2013 -f imagenet_vit_cos_tra-1
python calc_scores.py -d tid2013 -f imagenet_vit_l2_tra-1
python calc_scores.py -d tid2013 -f openclip_vitb32_laion_tecoa4_cos_tra-1
python calc_scores.py -d tid2013 -f embed_openclip_vitb32_laion_tecoa4_cos_tra-1
python calc_scores.py -d tid2013 -f openclip_convnext_laion_tecoa4_cos_tra-1
python calc_scores.py -d tid2013 -f embed_openclip_convnext_laion_tecoa4_cos_tra-1

echo "pipal"
# pipal
python calc_scores.py -d pipal -f imagenet_vit_cos_tra-1
python calc_scores.py -d pipal -f imagenet_vit_l2_tra-1
python calc_scores.py -d pipal -f openclip_vitb32_laion_tecoa4_cos_tra-1
python calc_scores.py -d pipal -f embed_openclip_vitb32_laion_tecoa4_cos_tra-1
python calc_scores.py -d pipal -f openclip_convnext_laion_tecoa4_cos_tra-1
python calc_scores.py -d pipal -f embed_openclip_convnext_laion_tecoa4_cos_tra-1

## sca
echo "live"
# live
python calc_scores.py -d live -f imagenet_vit_cos_sca-1
python calc_scores.py -d live -f imagenet_vit_l2_sca-1
python calc_scores.py -d live -f openclip_vitb32_laion_tecoa4_cos_sca-1
python calc_scores.py -d live -f embed_openclip_vitb32_laion_tecoa4_cos_sca-1
python calc_scores.py -d live -f openclip_convnext_laion_tecoa4_cos_sca-1
python calc_scores.py -d live -f embed_openclip_convnext_laion_tecoa4_cos_sca-1

echo "tid2013"
# tid2013
python calc_scores.py -d tid2013 -f imagenet_vit_cos_sca-1
python calc_scores.py -d tid2013 -f imagenet_vit_l2_sca-1
python calc_scores.py -d tid2013 -f openclip_vitb32_laion_tecoa4_cos_sca-1
python calc_scores.py -d tid2013 -f embed_openclip_vitb32_laion_tecoa4_cos_sca-1
python calc_scores.py -d tid2013 -f openclip_convnext_laion_tecoa4_cos_sca-1
python calc_scores.py -d tid2013 -f embed_openclip_convnext_laion_tecoa4_cos_sca-1

echo "pipal"
# pipal
python calc_scores.py -d pipal -f imagenet_vit_cos_sca-1
python calc_scores.py -d pipal -f imagenet_vit_l2_sca-1
python calc_scores.py -d pipal -f openclip_vitb32_laion_tecoa4_cos_sca-1
python calc_scores.py -d pipal -f embed_openclip_vitb32_laion_tecoa4_cos_sca-1
python calc_scores.py -d pipal -f openclip_convnext_laion_tecoa4_cos_sca-1
python calc_scores.py -d pipal -f embed_openclip_convnext_laion_tecoa4_cos_sca-1

## rot

echo "live"
# live
python calc_scores.py -d live -f imagenet_vit_cos_rot-1
python calc_scores.py -d live -f imagenet_vit_l2_rot-1
python calc_scores.py -d live -f openclip_vitb32_laion_tecoa4_cos_rot-1
python calc_scores.py -d live -f embed_openclip_vitb32_laion_tecoa4_cos_rot-1
python calc_scores.py -d live -f openclip_convnext_laion_tecoa4_cos_rot-1
python calc_scores.py -d live -f embed_openclip_convnext_laion_tecoa4_cos_rot-1

echo "tid2013"
# tid2013
python calc_scores.py -d tid2013 -f imagenet_vit_cos_rot-1
python calc_scores.py -d tid2013 -f imagenet_vit_l2_rot-1
python calc_scores.py -d tid2013 -f openclip_vitb32_laion_tecoa4_cos_rot-1
python calc_scores.py -d tid2013 -f embed_openclip_vitb32_laion_tecoa4_cos_rot-1
python calc_scores.py -d tid2013 -f openclip_convnext_laion_tecoa4_cos_rot-1
python calc_scores.py -d tid2013 -f embed_openclip_convnext_laion_tecoa4_cos_rot-1

echo "pipal"
# pipal
python calc_scores.py -d pipal -f imagenet_vit_cos_rot-1
python calc_scores.py -d pipal -f imagenet_vit_l2_rot-1
python calc_scores.py -d pipal -f openclip_vitb32_laion_tecoa4_cos_rot-1
python calc_scores.py -d pipal -f embed_openclip_vitb32_laion_tecoa4_cos_rot-1
python calc_scores.py -d pipal -f openclip_convnext_laion_tecoa4_cos_rot-1
python calc_scores.py -d pipal -f embed_openclip_convnext_laion_tecoa4_cos_rot-1