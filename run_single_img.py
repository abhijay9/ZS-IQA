import torch
import numpy as np
from utils import *
from datasets import *
from models import *
from distances import *

class ZS_IQA:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = self.init_model()

    def init_model(self):
        model = get_model(self.model_name, self.device)

        if "embed" not in self.model_name:
            def hook_fn(m, i, o):
                self.feats.append(o)

            if "clip" in self.model_name:
                if "vitb32" in self.model_name:
                    for i in range(12):
                        model.transformer.resblocks[i].register_forward_hook(hook_fn)
                elif "convnext" in self.model_name:
                    n_stages = len(model.trunk.stages)
                    for i in range(n_stages):
                        n_blocks = len(model.trunk.stages[i].blocks)
                        for j in range(n_blocks):
                            model.trunk.stages[i].blocks[j].register_forward_hook(hook_fn)
                elif "rn50" in self.model_name:
                    model.avgpool.register_forward_hook(hook_fn)
                    model.layer1.register_forward_hook(hook_fn)
                    model.layer2.register_forward_hook(hook_fn)
                    model.layer3.register_forward_hook(hook_fn)
                    model.layer4.register_forward_hook(hook_fn)

            elif "dino" in self.model_name:
                n_layers = len(model.encoder.layer)
                for i in range(n_layers):
                    model.encoder.layer[i].register_forward_hook(hook_fn)

            elif "imagenet_vit" in self.model_name:
                n_layers = len(model.vit.encoder.layer)
                for i in range(n_layers):
                    model.vit.encoder.layer[i].register_forward_hook(hook_fn)

        return model

    def get_score(self, ref, dis, distance="l2"):
        
        #  Calculate score for CLIP and DINO variants
        if "clip" in args.model or "dino" in args.model or "imagenet_vit" in args.model:
            pred_score = []
            _, _, h, w = dis.shape
            xs = get_indxs_sliding_window(length=h)
            ys = get_indxs_sliding_window(length=w)

            for x in xs:
                for y in ys:
                    if "openclip" in self.model_name or "dino" in self.model_name or "imagenet_vit" in self.model_name:
                        tensorType = torch.cuda.FloatTensor
                    elif "clip_vitb32" in self.model_name or "clip_rn50" in self.model_name:
                        tensorType = torch.cuda.HalfTensor
                    else:
                        raise ValueError("Incorrect model name")

                    # Store features in a list
                    self.feats = []
                    dis_out = self.model(dis.type(tensorType)[:, :, x:x+224, y:y+224])
                    feats_dis = self.feats

                    self.feats = []
                    ref_out = self.model(ref.type(tensorType)[:, :, x:x+224, y:y+224])
                    feats_ref = self.feats

                    # If not embedding then concat features
                    if "embed" not in self.model_name:
                        if "vitb32" in self.model_name:
                            feats_dis = torch.cat([feats_dis[i].detach().reshape([50,768]).unsqueeze(0) for i in range(12)])
                            feats_ref = torch.cat([feats_ref[i].detach().reshape([50,768]).unsqueeze(0) for i in range(12)])
                        
                            # batch processing for later...
                            feats_dis = feats_dis.unsqueeze(0)
                            feats_ref = feats_ref.unsqueeze(0)
                        elif "imagenet_vit" in self.model_name:
                            feats_dis = torch.cat([feats_dis[i][0] for i in range(len(feats_ref))])
                            feats_ref = torch.cat([feats_ref[i][0] for i in range(len(feats_ref))])
                            
                            # batch processing for later...
                            feats_dis = feats_dis.unsqueeze(0)
                            feats_ref = feats_ref.unsqueeze(0)
                        elif "dino" in self.model_name:
                            feats_dis = torch.cat([feats_dis[i][0] for i in range(len(feats_ref))])
                            feats_ref = torch.cat([feats_ref[i][0] for i in range(len(feats_ref))])
                            
                            # batch processing for later...
                            feats_dis = feats_dis.unsqueeze(0)
                            feats_ref = feats_ref.unsqueeze(0)
                        # skip for clip convnext and restnet versions


                    if args.distance == "l2":
                        if "convnext" in self.model_name or "rn50" in self.model_name:
                            score = np.sum([l2(feats_dis[i], feats_ref[i]) for i in range(len(feats_ref))])
                        else:
                            score = np.sum([l2(feats_dis[:,i], feats_ref[:,i]) for i in range(feats_ref.shape[1])])
                    elif args.distance == "swd":
                        score = np.sum(swd_dist(feats_dis, feats_ref, device))
                    elif args.distance == "cos":
                        if "embed" in self.model_name:
                            score = np.array(cos_dist(dis_out, ref_out))
                        elif "vitb32" in self.model_name or "dino" in self.model_name or "imagenet_vit" in self.model_name:
                            score = np.sum([cos_dist(feats_dis[:,i], feats_ref[:,i]) for i in range(feats_ref.shape[1])])
                        elif "convnext" in self.model_name:
                            score = np.sum([cos_dist(feats_dis[i], feats_ref[i]) for i in range(len(feats_ref))])
                        elif "rn50" in self.model_name:
                            score = np.sum([cos_dist(feats_dis[i], feats_ref[i], add_inf_handling=True) for i in range(len(feats_ref))])
                    elif args.distance == "skld":
                        window = 8
                        row_padding = round(feats_dis.size(2) / window) * window - feats_dis.size(2)
                        column_padding = round(feats_dis.size(3) / window) * window - feats_dis.size(3)
                        pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
                        feats_dis = pad(feats_dis)
                        feats_ref = pad(feats_ref)
                        score = KL_distance(feats_dis, feats_ref, win=window)
                        score = torch.log(score + 1)**0.25
                    elif args.distance == "wsd":
                        window = 8
                        row_padding = round(feats_dis.size(2) / window) * window - feats_dis.size(2)
                        column_padding = round(feats_dis.size(3) / window) * window - feats_dis.size(3)
                        pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
                        feats_dis = pad(feats_dis)
                        feats_ref = pad(feats_ref)
                        score = ws_distance(feats_dis, feats_ref, win=window, device=device)
                        score = torch.log(score + 1)**0.25
                    elif args.distance == "jsd":
                        window = 8
                        row_padding = round(feats_dis.size(2) / window) * window - feats_dis.size(2)
                        column_padding = round(feats_dis.size(3) / window) * window - feats_dis.size(3)
                        pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
                        feats_dis = pad(feats_dis)
                        feats_ref = pad(feats_ref)
                        score = js_distance(feats_dis, feats_ref, win=window)
                        score = torch.log(score + 1)**0.25
                    else:
                        print("incorrect model name")
                        raise ValueError

                    pred_score.append(score.item())
            d = np.mean(pred_score)
        else:
            d = model(dis, ref).item()

        return np.mean(pred_score)

# Usage example
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ref", '--ref_img', type=str, help='Reference image path')
    parser.add_argument("-dis", '--dis_img', type=str, help='Distorted image path')
    parser.add_argument("-m", '--model', type=str, default='clip_vitb32', help='Model to use')
    parser.add_argument("-d", '--distance', type=str, default='l2', help='Distance metric')
    parser.add_argument("-resizeHW", type=int, default=None, help='Resize height/width')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zs_iqa = ZS_IQA(args.model, device)

    img_transform = PrepareImg(args.resizeHW) if args.resizeHW else None
    prepare_image = img_transform.prepare_image if img_transform else prepare_image

    dis = prepare_image(load_img(args.dis_img).convert("RGB")).to(device)
    ref = prepare_image(load_img(args.ref_img).convert("RGB")).to(device)

    score = zs_iqa.get_score(ref, dis, distance=args.distance)
    print(f"Score: {score}")