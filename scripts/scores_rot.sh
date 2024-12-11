echo "\nlive"
python calc_scores.py -d live -f deep_kld_vgg_rot-1
python calc_scores.py -d live -f deep_jsd_vgg_rot-1
python calc_scores.py -d live -f deep_wsd_vgg_rot-1
python calc_scores.py -d live -f deep_kld_eff_rot-1
python calc_scores.py -d live -f deep_jsd_eff_rot-1
python calc_scores.py -d live -f deep_wsd_eff_rot-1

python calc_scores.py -d live -f clip_vitb32_l2_rot-1
python calc_scores.py -d live -f clip_vitb32_cos_rot-1
python calc_scores.py -d live -f clip_vitb32_swd_rot-1
python calc_scores.py -d live -f clip_vitb32_skld_rot-1
python calc_scores.py -d live -f clip_vitb32_wsd_rot-1
python calc_scores.py -d live -f clip_vitb32_jsd_rot-1

python calc_scores.py -d live -f psnr_rot-1
python calc_scores.py -d live -f msssim_rot-1
python calc_scores.py -d live -f fsim_rot-1
python calc_scores.py -d live -f dists_rot-1
python calc_scores.py -d live -f lpips_rot-1
python calc_scores.py -d live -f lpips_vgg_rot-1
python calc_scores.py -d live -f stlpips_alex_rot-1
python calc_scores.py -d live -f stlpips_vgg_rot-1
python calc_scores.py -d live -f lpips_vgg_kadid_rot-1
python calc_scores.py -d live -f stlpips_vgg_kadid_rot-1


echo "\ntid2013"
python calc_scores.py -d tid2013 -f deep_kld_vgg_rot-1
python calc_scores.py -d tid2013 -f deep_jsd_vgg_rot-1
python calc_scores.py -d tid2013 -f deep_wsd_vgg_rot-1
python calc_scores.py -d tid2013 -f deep_kld_eff_rot-1
python calc_scores.py -d tid2013 -f deep_jsd_eff_rot-1
python calc_scores.py -d tid2013 -f deep_wsd_eff_rot-1

python calc_scores.py -d tid2013 -f clip_vitb32_l2_rot-1
python calc_scores.py -d tid2013 -f clip_vitb32_cos_rot-1
python calc_scores.py -d tid2013 -f clip_vitb32_swd_rot-1
python calc_scores.py -d tid2013 -f clip_vitb32_skld_rot-1
python calc_scores.py -d tid2013 -f clip_vitb32_wsd_rot-1
python calc_scores.py -d tid2013 -f clip_vitb32_jsd_rot-1

python calc_scores.py -d tid2013 -f psnr_rot-1
python calc_scores.py -d tid2013 -f msssim_rot-1
python calc_scores.py -d tid2013 -f fsim_rot-1
python calc_scores.py -d tid2013 -f dists_rot-1
python calc_scores.py -d tid2013 -f lpips_rot-1
python calc_scores.py -d tid2013 -f lpips_vgg_rot-1
python calc_scores.py -d tid2013 -f stlpips_alex_rot-1
python calc_scores.py -d tid2013 -f stlpips_vgg_rot-1
python calc_scores.py -d tid2013 -f lpips_vgg_kadid_rot-1
python calc_scores.py -d tid2013 -f stlpips_vgg_kadid_rot-1

echo "\npipal"
python calc_scores.py -d pipal -f deep_kld_vgg_rot-1
python calc_scores.py -d pipal -f deep_jsd_vgg_rot-1
python calc_scores.py -d pipal -f deep_wsd_vgg_rot-1
python calc_scores.py -d pipal -f deep_kld_eff_rot-1
python calc_scores.py -d pipal -f deep_jsd_eff_rot-1
python calc_scores.py -d pipal -f deep_wsd_eff_rot-1

python calc_scores.py -d pipal -f clip_vitb32_l2_rot-1
python calc_scores.py -d pipal -f clip_vitb32_cos_rot-1
python calc_scores.py -d pipal -f clip_vitb32_swd_rot-1
python calc_scores.py -d pipal -f clip_vitb32_skld_rot-1
python calc_scores.py -d pipal -f clip_vitb32_wsd_rot-1
python calc_scores.py -d pipal -f clip_vitb32_jsd_rot-1

python calc_scores.py -d pipal -f psnr_rot-1
python calc_scores.py -d pipal -f msssim_rot-1
python calc_scores.py -d pipal -f fsim_rot-1
python calc_scores.py -d pipal -f dists_rot-1
python calc_scores.py -d pipal -f lpips_rot-1
python calc_scores.py -d pipal -f lpips_vgg_rot-1
python calc_scores.py -d pipal -f stlpips_alex_rot-1
python calc_scores.py -d pipal -f stlpips_vgg_rot-1
python calc_scores.py -d pipal -f lpips_vgg_kadid_rot-1
python calc_scores.py -d pipal -f stlpips_vgg_kadid_rot-1
