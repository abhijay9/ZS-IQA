# live
python run.py -dat live -d l2 -m dinov1 -s dinov1_l2
python run.py -dat live -d cos -m dinov1 -s dinov1_cos
python run.py -dat live -d skld -m dinov1 -s dinov1_skld
python run.py -dat live -d wsd -m dinov1 -s dinov1_wsd
python run.py -dat live -d jsd -m dinov1 -s dinov1_jsd
python run.py -dat live -d swd -m dinov1 -s dinov1_swd
python run.py -dat live -d cos -m openclip_convnext_base -s openclip_convnext_base_cos
python run.py -dat live -d cos -m dinov2 -s dinov2_cos
python run.py -dat live -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos
python run.py -dat live -d cos -m embed_dinov1 -s embed_dinov1_cos
python run.py -dat live -d cos -m embed_dinov2 -s embed_dinov2_cos
python run.py -dat live -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos
python run.py -dat live -d cos -m clip_rn50 -s clip_rn50_cos
python run.py -dat live -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos
python run.py -dat live -d l2 -m clip_rn50 -s clip_rn50_l2
python run.py -dat live -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2

# tid2013
python run.py -dat tid2013 -d l2 -m dinov1 -s dinov1_l2
python run.py -dat tid2013 -d cos -m dinov1 -s dinov1_cos
python run.py -dat tid2013 -d skld -m dinov1 -s dinov1_skld
python run.py -dat tid2013 -d wsd -m dinov1 -s dinov1_wsd
python run.py -dat tid2013 -d jsd -m dinov1 -s dinov1_jsd
python run.py -dat tid2013 -d swd -m dinov1 -s dinov1_swd
python run.py -dat tid2013 -d cos -m openclip_convnext_base -s openclip_convnext_base_cos
python run.py -dat tid2013 -d cos -m dinov2 -s dinov2_cos
python run.py -dat tid2013 -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos
python run.py -dat tid2013 -d cos -m embed_dinov1 -s embed_dinov1_cos
python run.py -dat tid2013 -d cos -m embed_dinov2 -s embed_dinov2_cos
python run.py -dat tid2013 -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos
python run.py -dat tid2013 -d cos -m clip_rn50 -s clip_rn50_cos
python run.py -dat tid2013 -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos
python run.py -dat tid2013 -d l2 -m clip_rn50 -s clip_rn50_l2
python run.py -dat tid2013 -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2

# pipal
python run.py -dat pipal -d l2 -m dinov1 -s dinov1_l2
python run.py -dat pipal -d cos -m dinov1 -s dinov1_cos
python run.py -dat pipal -d skld -m dinov1 -s dinov1_skld
python run.py -dat pipal -d wsd -m dinov1 -s dinov1_wsd
python run.py -dat pipal -d jsd -m dinov1 -s dinov1_jsd
python run.py -dat pipal -d swd -m dinov1 -s dinov1_swd
python run.py -dat pipal -d cos -m openclip_convnext_base -s openclip_convnext_base_cos
python run.py -dat pipal -d cos -m dinov2 -s dinov2_cos
python run.py -dat pipal -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos
python run.py -dat pipal -d cos -m embed_dinov1 -s embed_dinov1_cos
python run.py -dat pipal -d cos -m embed_dinov2 -s embed_dinov2_cos
python run.py -dat pipal -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos
python run.py -dat pipal -d cos -m clip_rn50 -s clip_rn50_cos
python run.py -dat pipal -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos
python run.py -dat pipal -d l2 -m clip_rn50 -s clip_rn50_l2
python run.py -dat pipal -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2
python run.py -dat pipal -d l2 -m dinov2 -s dinov2_l2



##### robustness


## live
python run.py -dat live -d l2 -m dinov1 -s dinov1_l2 -rob tra
python run.py -dat live -d cos -m dinov1 -s dinov1_cos -rob tra
python run.py -dat live -d skld -m dinov1 -s dinov1_skld -rob tra
python run.py -dat live -d wsd -m dinov1 -s dinov1_wsd -rob tra
python run.py -dat live -d jsd -m dinov1 -s dinov1_jsd -rob tra
python run.py -dat live -d swd -m dinov1 -s dinov1_swd -rob tra
python run.py -dat live -d cos -m openclip_convnext_base -s openclip_convnext_base_cos -rob tra
python run.py -dat live -d cos -m dinov2 -s dinov2_cos -rob tra
python run.py -dat live -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos -rob tra
python run.py -dat live -d cos -m embed_dinov1 -s embed_dinov1_cos -rob tra
python run.py -dat live -d cos -m embed_dinov2 -s embed_dinov2_cos -rob tra
python run.py -dat live -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos -rob tra
python run.py -dat live -d cos -m clip_rn50 -s clip_rn50_cos -rob tra
python run.py -dat live -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos -rob tra
python run.py -dat live -d l2 -m clip_rn50 -s clip_rn50_l2 -rob tra
python run.py -dat live -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2 -rob tra

#sca
python run.py -dat live -d l2 -m dinov1 -s dinov1_l2 -rob sca
python run.py -dat live -d cos -m dinov1 -s dinov1_cos -rob sca
python run.py -dat live -d skld -m dinov1 -s dinov1_skld -rob sca
python run.py -dat live -d wsd -m dinov1 -s dinov1_wsd -rob sca
python run.py -dat live -d jsd -m dinov1 -s dinov1_jsd -rob sca
python run.py -dat live -d swd -m dinov1 -s dinov1_swd -rob sca
python run.py -dat live -d cos -m openclip_convnext_base -s openclip_convnext_base_cos -rob sca
python run.py -dat live -d cos -m dinov2 -s dinov2_cos -rob sca
python run.py -dat live -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos -rob sca
python run.py -dat live -d cos -m embed_dinov1 -s embed_dinov1_cos -rob sca
python run.py -dat live -d cos -m embed_dinov2 -s embed_dinov2_cos -rob sca
python run.py -dat live -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos -rob sca
python run.py -dat live -d cos -m clip_rn50 -s clip_rn50_cos -rob sca
python run.py -dat live -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos -rob sca
python run.py -dat live -d l2 -m clip_rn50 -s clip_rn50_l2 -rob sca
python run.py -dat live -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2 -rob sca

#rot
python run.py -dat live -d l2 -m dinov1 -s dinov1_l2 -rob rot
python run.py -dat live -d cos -m dinov1 -s dinov1_cos -rob rot
python run.py -dat live -d skld -m dinov1 -s dinov1_skld -rob rot
python run.py -dat live -d wsd -m dinov1 -s dinov1_wsd -rob rot
python run.py -dat live -d jsd -m dinov1 -s dinov1_jsd -rob rot
python run.py -dat live -d swd -m dinov1 -s dinov1_swd -rob rot
python run.py -dat live -d cos -m openclip_convnext_base -s openclip_convnext_base_cos -rob rot
python run.py -dat live -d cos -m dinov2 -s dinov2_cos -rob rot
python run.py -dat live -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos -rob rot
python run.py -dat live -d cos -m embed_dinov1 -s embed_dinov1_cos -rob rot
python run.py -dat live -d cos -m embed_dinov2 -s embed_dinov2_cos -rob rot
python run.py -dat live -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos -rob rot
python run.py -dat live -d cos -m clip_rn50 -s clip_rn50_cos -rob rot
python run.py -dat live -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos -rob rot
python run.py -dat live -d l2 -m clip_rn50 -s clip_rn50_l2 -rob rot
python run.py -dat live -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2 -rob rot



## tid2013
python run.py -dat tid2013 -d l2 -m dinov1 -s dinov1_l2 -rob tra
python run.py -dat tid2013 -d cos -m dinov1 -s dinov1_cos -rob tra
python run.py -dat tid2013 -d skld -m dinov1 -s dinov1_skld -rob tra
python run.py -dat tid2013 -d wsd -m dinov1 -s dinov1_wsd -rob tra
python run.py -dat tid2013 -d jsd -m dinov1 -s dinov1_jsd -rob tra
python run.py -dat tid2013 -d swd -m dinov1 -s dinov1_swd -rob tra
python run.py -dat tid2013 -d cos -m openclip_convnext_base -s openclip_convnext_base_cos -rob tra
python run.py -dat tid2013 -d cos -m dinov2 -s dinov2_cos -rob tra
python run.py -dat tid2013 -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos -rob tra
python run.py -dat tid2013 -d cos -m embed_dinov1 -s embed_dinov1_cos -rob tra
python run.py -dat tid2013 -d cos -m embed_dinov2 -s embed_dinov2_cos -rob tra
python run.py -dat tid2013 -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos -rob tra
python run.py -dat tid2013 -d cos -m clip_rn50 -s clip_rn50_cos -rob tra
python run.py -dat tid2013 -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos -rob tra
python run.py -dat tid2013 -d l2 -m clip_rn50 -s clip_rn50_l2 -rob tra
python run.py -dat tid2013 -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2 -rob tra

#sca
python run.py -dat tid2013 -d l2 -m dinov1 -s dinov1_l2 -rob sca
python run.py -dat tid2013 -d cos -m dinov1 -s dinov1_cos -rob sca
python run.py -dat tid2013 -d skld -m dinov1 -s dinov1_skld -rob sca
python run.py -dat tid2013 -d wsd -m dinov1 -s dinov1_wsd -rob sca
python run.py -dat tid2013 -d jsd -m dinov1 -s dinov1_jsd -rob sca
python run.py -dat tid2013 -d swd -m dinov1 -s dinov1_swd -rob sca
python run.py -dat tid2013 -d cos -m openclip_convnext_base -s openclip_convnext_base_cos -rob sca
python run.py -dat tid2013 -d cos -m dinov2 -s dinov2_cos -rob sca
python run.py -dat tid2013 -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos -rob sca
python run.py -dat tid2013 -d cos -m embed_dinov1 -s embed_dinov1_cos -rob sca
python run.py -dat tid2013 -d cos -m embed_dinov2 -s embed_dinov2_cos -rob sca
python run.py -dat tid2013 -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos -rob sca
python run.py -dat tid2013 -d cos -m clip_rn50 -s clip_rn50_cos -rob sca
python run.py -dat tid2013 -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos -rob sca
python run.py -dat tid2013 -d l2 -m clip_rn50 -s clip_rn50_l2 -rob sca
python run.py -dat tid2013 -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2 -rob sca

#rot
python run.py -dat tid2013 -d l2 -m dinov1 -s dinov1_l2 -rob rot
python run.py -dat tid2013 -d cos -m dinov1 -s dinov1_cos -rob rot
python run.py -dat tid2013 -d skld -m dinov1 -s dinov1_skld -rob rot
python run.py -dat tid2013 -d wsd -m dinov1 -s dinov1_wsd -rob rot
python run.py -dat tid2013 -d jsd -m dinov1 -s dinov1_jsd -rob rot
python run.py -dat tid2013 -d swd -m dinov1 -s dinov1_swd -rob rot
python run.py -dat tid2013 -d cos -m openclip_convnext_base -s openclip_convnext_base_cos -rob rot
python run.py -dat tid2013 -d cos -m dinov2 -s dinov2_cos -rob rot
python run.py -dat tid2013 -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos -rob rot
python run.py -dat tid2013 -d cos -m embed_dinov1 -s embed_dinov1_cos -rob rot
python run.py -dat tid2013 -d cos -m embed_dinov2 -s embed_dinov2_cos -rob rot
python run.py -dat tid2013 -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos -rob rot
python run.py -dat tid2013 -d cos -m clip_rn50 -s clip_rn50_cos -rob rot
python run.py -dat tid2013 -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos -rob rot
python run.py -dat tid2013 -d l2 -m clip_rn50 -s clip_rn50_l2 -rob rot
python run.py -dat tid2013 -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2 -rob rot



## pipal
python run.py -dat pipal -d l2 -m dinov1 -s dinov1_l2 -rob tra
python run.py -dat pipal -d cos -m dinov1 -s dinov1_cos -rob tra
python run.py -dat pipal -d skld -m dinov1 -s dinov1_skld -rob tra
python run.py -dat pipal -d wsd -m dinov1 -s dinov1_wsd -rob tra
python run.py -dat pipal -d jsd -m dinov1 -s dinov1_jsd -rob tra
python run.py -dat pipal -d swd -m dinov1 -s dinov1_swd -rob tra
python run.py -dat pipal -d cos -m openclip_convnext_base -s openclip_convnext_base_cos -rob tra
python run.py -dat pipal -d cos -m dinov2 -s dinov2_cos -rob tra
python run.py -dat pipal -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos -rob tra
python run.py -dat pipal -d cos -m embed_dinov1 -s embed_dinov1_cos -rob tra
python run.py -dat pipal -d cos -m embed_dinov2 -s embed_dinov2_cos -rob tra
python run.py -dat pipal -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos -rob tra
python run.py -dat pipal -d cos -m clip_rn50 -s clip_rn50_cos -rob tra
python run.py -dat pipal -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos -rob tra
python run.py -dat pipal -d l2 -m clip_rn50 -s clip_rn50_l2 -rob tra
python run.py -dat pipal -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2 -rob tra
python run.py -dat pipal -d l2 -m dinov2 -s dinov2_l2 -rob tra

#sca
python run.py -dat pipal -d l2 -m dinov1 -s dinov1_l2 -rob sca
python run.py -dat pipal -d cos -m dinov1 -s dinov1_cos -rob sca
python run.py -dat pipal -d skld -m dinov1 -s dinov1_skld -rob sca
python run.py -dat pipal -d wsd -m dinov1 -s dinov1_wsd -rob sca
python run.py -dat pipal -d jsd -m dinov1 -s dinov1_jsd -rob sca
python run.py -dat pipal -d swd -m dinov1 -s dinov1_swd -rob sca
python run.py -dat pipal -d cos -m openclip_convnext_base -s openclip_convnext_base_cos -rob sca
python run.py -dat pipal -d cos -m dinov2 -s dinov2_cos -rob sca
python run.py -dat pipal -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos -rob sca
python run.py -dat pipal -d cos -m embed_dinov1 -s embed_dinov1_cos -rob sca
python run.py -dat pipal -d cos -m embed_dinov2 -s embed_dinov2_cos -rob sca
python run.py -dat pipal -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos -rob sca
python run.py -dat pipal -d cos -m clip_rn50 -s clip_rn50_cos -rob sca
python run.py -dat pipal -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos -rob sca
python run.py -dat pipal -d l2 -m clip_rn50 -s clip_rn50_l2 -rob sca
python run.py -dat pipal -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2 -rob sca
python run.py -dat pipal -d l2 -m dinov2 -s dinov2_l2 -rob sca

#rot
python run.py -dat pipal -d l2 -m dinov1 -s dinov1_l2 -rob rot
python run.py -dat pipal -d cos -m dinov1 -s dinov1_cos -rob rot
python run.py -dat pipal -d skld -m dinov1 -s dinov1_skld -rob rot
python run.py -dat pipal -d wsd -m dinov1 -s dinov1_wsd -rob rot
python run.py -dat pipal -d jsd -m dinov1 -s dinov1_jsd -rob rot
python run.py -dat pipal -d swd -m dinov1 -s dinov1_swd -rob rot
python run.py -dat pipal -d cos -m openclip_convnext_base -s openclip_convnext_base_cos -rob rot
python run.py -dat pipal -d cos -m dinov2 -s dinov2_cos -rob rot
python run.py -dat pipal -d cos -m embed_clip_vitb32 -s embed_clip_vitb32_cos -rob rot
python run.py -dat pipal -d cos -m embed_dinov1 -s embed_dinov1_cos -rob rot
python run.py -dat pipal -d cos -m embed_dinov2 -s embed_dinov2_cos -rob rot
python run.py -dat pipal -d cos -m embed_openclip_convnext_base -s embed_openclip_convnext_base_cos -rob rot
python run.py -dat pipal -d cos -m clip_rn50 -s clip_rn50_cos -rob rot
python run.py -dat pipal -d cos -m embed_clip_rn50 -s embed_clip_rn50_cos -rob rot
python run.py -dat pipal -d l2 -m clip_rn50 -s clip_rn50_l2 -rob rot
python run.py -dat pipal -d l2 -m openclip_convnext_base -s openclip_convnext_base_l2 -rob rot
python run.py -dat pipal -d l2 -m dinov2 -s dinov2_l2 -rob rot