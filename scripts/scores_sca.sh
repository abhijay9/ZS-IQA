echo "\nlive"
python calc_scores.py -d live -f deep_kld_vgg_sca-1
python calc_scores.py -d live -f deep_jsd_vgg_sca-1
python calc_scores.py -d live -f deep_wsd_vgg_sca-1
python calc_scores.py -d live -f deep_kld_eff_sca-1
python calc_scores.py -d live -f deep_jsd_eff_sca-1
python calc_scores.py -d live -f deep_wsd_eff_sca-1

python calc_scores.py -d live -f clip_vitb32_l2_sca-1
python calc_scores.py -d live -f clip_vitb32_cos_sca-1
python calc_scores.py -d live -f clip_vitb32_swd_sca-1
python calc_scores.py -d live -f clip_vitb32_skld_sca-1
python calc_scores.py -d live -f clip_vitb32_wsd_sca-1
python calc_scores.py -d live -f clip_vitb32_jsd_sca-1

python calc_scores.py -d live -f psnr_sca-1
python calc_scores.py -d live -f msssim_sca-1
python calc_scores.py -d live -f fsim_sca-1
python calc_scores.py -d live -f dists_sca-1
python calc_scores.py -d live -f lpips_sca-1
python calc_scores.py -d live -f lpips_vgg_sca-1
python calc_scores.py -d live -f stlpips_alex_sca-1
python calc_scores.py -d live -f stlpips_vgg_sca-1
python calc_scores.py -d live -f lpips_vgg_kadid_sca-1
python calc_scores.py -d live -f stlpips_vgg_kadid_sca-1


echo "\ntid2013"
python calc_scores.py -d tid2013 -f deep_kld_vgg_sca-1
python calc_scores.py -d tid2013 -f deep_jsd_vgg_sca-1
python calc_scores.py -d tid2013 -f deep_wsd_vgg_sca-1
python calc_scores.py -d tid2013 -f deep_kld_eff_sca-1
python calc_scores.py -d tid2013 -f deep_jsd_eff_sca-1
python calc_scores.py -d tid2013 -f deep_wsd_eff_sca-1

python calc_scores.py -d tid2013 -f clip_vitb32_l2_sca-1
python calc_scores.py -d tid2013 -f clip_vitb32_cos_sca-1
python calc_scores.py -d tid2013 -f clip_vitb32_swd_sca-1
python calc_scores.py -d tid2013 -f clip_vitb32_skld_sca-1
python calc_scores.py -d tid2013 -f clip_vitb32_wsd_sca-1
python calc_scores.py -d tid2013 -f clip_vitb32_jsd_sca-1

python calc_scores.py -d tid2013 -f psnr_sca-1
python calc_scores.py -d tid2013 -f msssim_sca-1
python calc_scores.py -d tid2013 -f fsim_sca-1
python calc_scores.py -d tid2013 -f dists_sca-1
python calc_scores.py -d tid2013 -f lpips_sca-1
python calc_scores.py -d tid2013 -f lpips_vgg_sca-1
python calc_scores.py -d tid2013 -f stlpips_alex_sca-1
python calc_scores.py -d tid2013 -f stlpips_vgg_sca-1
python calc_scores.py -d tid2013 -f lpips_vgg_kadid_sca-1
python calc_scores.py -d tid2013 -f stlpips_vgg_kadid_sca-1

echo "\npipal"
python calc_scores.py -d pipal -f deep_kld_vgg_sca-1
python calc_scores.py -d pipal -f deep_jsd_vgg_sca-1
python calc_scores.py -d pipal -f deep_wsd_vgg_sca-1
python calc_scores.py -d pipal -f deep_kld_eff_sca-1
python calc_scores.py -d pipal -f deep_jsd_eff_sca-1
python calc_scores.py -d pipal -f deep_wsd_eff_sca-1

python calc_scores.py -d pipal -f clip_vitb32_l2_sca-1
python calc_scores.py -d pipal -f clip_vitb32_cos_sca-1
python calc_scores.py -d pipal -f clip_vitb32_swd_sca-1
python calc_scores.py -d pipal -f clip_vitb32_skld_sca-1
python calc_scores.py -d pipal -f clip_vitb32_wsd_sca-1
python calc_scores.py -d pipal -f clip_vitb32_jsd_sca-1

python calc_scores.py -d pipal -f psnr_sca-1
python calc_scores.py -d pipal -f msssim_sca-1
python calc_scores.py -d pipal -f fsim_sca-1
python calc_scores.py -d pipal -f dists_sca-1
python calc_scores.py -d pipal -f lpips_sca-1
python calc_scores.py -d pipal -f lpips_vgg_sca-1
python calc_scores.py -d pipal -f stlpips_alex_sca-1
python calc_scores.py -d pipal -f stlpips_vgg_sca-1
python calc_scores.py -d pipal -f lpips_vgg_kadid_sca-1
python calc_scores.py -d pipal -f stlpips_vgg_kadid_sca-1
