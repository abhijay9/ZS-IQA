import torch
import clip
import open_clip
import torch.nn.functional as F

# def normalize_embedding(embed):
#     if len(embed.shape) <= 1:
#         embed = embed.unsqueeze(0)
#     embed = (embed.T / torch.norm(embed, dim=1)).T
#     return (embed.T - torch.mean(embed, dim=1)).T

def get_model(model_name, device):
    if model_name == "clip_vitb32":
        model, transform = clip.load("ViT-B/32", jit=False, device=device)
        return model.visual
    elif model_name == "embed_clip_vitb32":
        model, transform = clip.load("ViT-B/32", jit=False, device=device)
        def model_forward(inp):
            embedding = model.encode_image(inp)
            # import pdb; pdb.set_trace()
            return F.normalize(embedding, dim=-1)
        return model_forward
    if model_name == "clip_rn50":
        model, transform = clip.load("RN50", jit=False, device=device)
        return model.visual
    elif model_name == "embed_clip_rn50":
        model, transform = clip.load("RN50", jit=False, device=device)
        def model_forward(inp):
            embedding = model.encode_image(inp)
            # import pdb; pdb.set_trace()
            return F.normalize(embedding, dim=-1)
        return model_forward
    elif model_name == "openclip_vitb32_laion":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
        return model.visual
    elif model_name == "openclip_vitb32_laion_tecoa4":
        model, _, image_processor = open_clip.create_model_and_transforms('hf-hub:chs20/TeCoA4-ViT-B-32-laion2B-s34B-b79K', device=device)
        return model.visual
    elif model_name == "embed_openclip_vitb32_laion_tecoa4":
        model, _, image_processor = open_clip.create_model_and_transforms('hf-hub:chs20/TeCoA4-ViT-B-32-laion2B-s34B-b79K', device=device)
        def model_forward(inp):
            embedding = model.encode_image(inp)
            return F.normalize(embedding, dim=-1)
        return model_forward
    elif model_name == "openclip_convnext_laion_tecoa4":
        model, _, image_processor = open_clip.create_model_and_transforms('hf-hub:chs20/TeCoA4-convnext_base_w-laion2B-s13B-b82K-augreg', device=device)
        return model.visual
    elif model_name == "embed_openclip_convnext_laion_tecoa4":
        model, _, image_processor = open_clip.create_model_and_transforms('hf-hub:chs20/TeCoA4-convnext_base_w-laion2B-s13B-b82K-augreg', device=device)
        def model_forward(inp):
            embedding = model.encode_image(inp)
            return F.normalize(embedding, dim=-1)
        return model_forward
    elif model_name == "openclip_convnext_base":
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K', device=device)
        return model.visual
    elif model_name == "embed_openclip_convnext_base":
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K', device=device)
        def model_forward(inp):
            embedding = model.encode_image(inp)
            return F.normalize(embedding, dim=-1)
        return model_forward
    if model_name == "imagenet_vit":
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
        return model
    # elif model_name == "embed_imagenet_vit":
    #     from transformers import ViTForImageClassification
    #     model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
    #     def model_forward(inp):
    #         outputs = model(inp)
    #         import pdb; pdb.set_trace()
    #         last_hidden_states = outputs.last_hidden_state
    #         # import pdb; pdb.set_trace()
    #         return F.normalize(last_hidden_states, dim=-1)
    #     return model_forward
    elif model_name == "dinov1":
        from transformers import ViTModel
        model = ViTModel.from_pretrained('facebook/dino-vitb16').to(device)
        return model
    elif model_name == "embed_dinov1":
        from transformers import ViTModel
        model = ViTModel.from_pretrained('facebook/dino-vitb16').to(device)
        def model_forward(inp):
            outputs = model(inp)
            last_hidden_states = outputs.last_hidden_state
            # import pdb; pdb.set_trace()
            return F.normalize(last_hidden_states, dim=-1)
        return model_forward
    elif model_name == "dinov2":
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        return model
    elif model_name == "embed_dinov2":
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        def model_forward(inp):
            outputs = model(inp)
            last_hidden_states = outputs.last_hidden_state
            # import pdb; pdb.set_trace()
            return F.normalize(last_hidden_states, dim=-1)
        return model_forward
    elif model_name == "fsim":
        import pyiqa
        D = pyiqa.create_metric('fsim', device=device)
        def calc_fsim(ref,dis):
            return (1. - D(ref, dis)) / 2.
        return calc_fsim
    elif model_name == "msssim":
        import pyiqa
        D = pyiqa.create_metric('ms_ssim', device=device)
        def calc_msssim(ref,dis):
            return (1. - D(ref, dis)) / 2.
        return calc_msssim
    elif model_name == "psnr":
        import pyiqa
        return pyiqa.create_metric('psnr', device=device)
    elif model_name == "dists":
        from DISTS_pytorch import DISTS
        return DISTS().to(device=device)
    # elif model_name == "mad":
    #     import pyiqa
    #     return pyiqa.create_metric('mad', device=device)
    # elif model_name == "wadiqam":
    #     import pyiqa
    #     return  pyiqa.create_metric('wadiqam_fr', device=device)
    elif "lpips" in model_name:
        if model_name == "stlpips_alex":
            import pyiqa
            return  pyiqa.create_metric('stlpips', device=device)
        elif model_name == "stlpips_vgg":
            import pyiqa
            return  pyiqa.create_metric('stlpips-vgg', device=device)
        elif model_name == "stlpips_vgg_kadid":
            import pyiqa
            return  pyiqa.create_metric('stlpips-vgg-kadid', device=device)
        elif model_name == "stlpips_vgg_kadid":
            import pyiqa
            return  pyiqa.create_metric('stlpips-vgg-kadid', device=device)
        elif model_name == "lpips_vgg_kadid":
            import pyiqa
            return  pyiqa.create_metric('lpips-vgg-kadid', device=device)
        elif model_name == "lpips":
            import lpips
            return  lpips.LPIPS(net='alex').to(device=device)
        elif model_name == "lpips_vgg":
            import lpips
            return  lpips.LPIPS(net='vgg').to(device=device)
    elif "deep_" in model_name:
        import sys
        sys.path.append("/home/abhijay/Documents/work/clip_test/")
        if "_kld_vgg" in model_name:
            from DeepDistanceMeasures.DeepKLD_VGG import DeepKLD
            model = DeepKLD().to(device)
        elif "_jsd_vgg" in model_name:
            from DeepDistanceMeasures.DeepJSD_VGG import DeepJSD
            model = DeepJSD().to(device)
        elif "_wsd_vgg" in model_name:
            from DeepDistanceMeasures.DeepWSD_VGG import DeepWSD
            model = DeepWSD().to(device)
        elif "_kld_eff" in model_name:
            from DeepDistanceMeasures.DeepKLD_Efficient import DeepKLD_eff
            model = DeepKLD_eff().to(device)
        elif "_jsd_eff" in model_name:
            from DeepDistanceMeasures.DeepJSD_Efficient import DeepJSD_eff
            model = DeepJSD_eff().to(device)
        elif "_wsd_eff" in model_name:
            from DeepDistanceMeasures.DeepWSD_Efficient import DeepWSD_eff
            model = DeepWSD_eff().to(device)
        def calc_score(ref,dis):
            return model(ref, dis, as_loss=False, resize=False)
        
        return calc_score

    # elif args.model == "pieapp":
    #     print("pieapp")
    # elif model_name == "openclip_vitb32_laion_tecoa4":
    #     model, _, image_processor = open_clip.create_model_and_transforms('hf-hub:chs20/TeCoA4-ViT-B-32-laion2B-s34B-b79K', device=device)
    #     return model
    
    else:
        ValueError