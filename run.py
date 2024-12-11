from tqdm import tqdm
import argparse
import random
import os

from utils import *
from datasets import *
from models import *
from distances import *

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dat", '--dataset', type=str, default='live', help='datasets to test on: [live],[tid2013],[pipal]')
    parser.add_argument("-m", '--model', type=str, default='clip_vitb32', help='models to test: [clip_vitb32], [clip_rn50], ')
    parser.add_argument("-d", '--distance', type=str, default='l2', help='models to test: [l2], [swd], ')
    parser.add_argument("-s", '--saveas', type=str, default='save', help='save results in filename')
    parser.add_argument("-sdir", '--savedir', type=str, default='results', help='save results dir')
    parser.add_argument("-r", '--resizeHW', type=int, default=None, help='most common value is 256')
    parser.add_argument("-rob", '--robustness', type=str, default=None, help='save results in filename')
    parser.add_argument("-pct", '--pctPixels', type=int, default=1, help='save results in filename')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = get_dataset(args.dataset)

    model = get_model(args.model, device)

    os.makedirs(args.savedir, exist_ok=True) 
    if args.robustness:
        f = open(args.savedir+"/"+args.dataset+"_"+args.saveas+"_"+args.robustness+"-"+str(args.pctPixels)+".csv","w")
    else:
        f = open(args.savedir+"/"+args.dataset+"_"+args.saveas+".csv","w")
    f.write("dis,ref,mos,pred\n")

    if "embed" not in args.model:
        
        def hook_fn(m, i, o):
            feats.append(o)

        # CLIP variants
        if "clip" in args.model:
            if "vitb32" in args.model:
                for i in range(12):
                    model.transformer.resblocks[i].register_forward_hook(hook_fn)
            elif "convnext" in args.model:
                n_stages = len(model.trunk.stages)
                for i in range(n_stages):
                    n_blocks = len(model.trunk.stages[i].blocks)
                    for j in range(n_blocks):
                        model.trunk.stages[i].blocks[j].register_forward_hook(hook_fn)
            elif "rn50" in args.model:
                model.avgpool.register_forward_hook(hook_fn)
                model.layer1.register_forward_hook(hook_fn)
                model.layer2.register_forward_hook(hook_fn)
                model.layer3.register_forward_hook(hook_fn)
                model.layer4.register_forward_hook(hook_fn)
        
        # DINO variants
        elif "dino" in args.model:
            n_layers = len(model.encoder.layer)
            for i in range(n_layers):
                model.encoder.layer[i].register_forward_hook(hook_fn)
        
        elif "imagenet_vit" in args.model:
            n_layers = len(model.vit.encoder.layer)
            for i in range(n_layers):
                model.vit.encoder.layer[i].register_forward_hook(hook_fn)

    for data_i in tqdm(data):
        
        # Add distortion to image based on test type: [original or robustness]
        if args.resizeHW:
            img_transform = PrepareImg(args.resizeHW)
            prepare_image = img_transform.prepare_image

        if args.robustness:
            dis = prepare_image(load_img(data_i["dis_im"], robustness_distortion=args.robustness, pct_shift=args.pctPixels, ref=False).convert("RGB")).to(device=device)
            ref = prepare_image(load_img(data_i["ref_im"], robustness_distortion=args.robustness, pct_shift=args.pctPixels, ref=True).convert("RGB")).to(device=device)
        else:
            dis = prepare_image(load_img(data_i["dis_im"]).convert("RGB")).to(device=device)
            ref = prepare_image(load_img(data_i["ref_im"]).convert("RGB")).to(device=device)

        #  Calculate score for CLIP and DINO variants
        if "clip" in args.model or "dino" in args.model or "imagenet_vit" in args.model:
            
            pred_score = []

            # Sliding window operation
            _, _, h, w = dis.shape
            xs = get_indxs_sliding_window(length=h)
            ys = get_indxs_sliding_window(length=w)
            for x in xs:
                for y in ys:
                    # Assign tensor type based on prec of model
                    if "openclip" in args.model or "dino" in args.model or "imagenet_vit" in args.model:
                        tensorType = torch.cuda.FloatTensor
                    elif "clip_vitb32" in args.model:
                        tensorType = torch.cuda.HalfTensor
                    elif "clip_rn50" in args.model:
                        tensorType = torch.cuda.HalfTensor
                    else:
                        print("incorrect model name")
                        raise ValueError
                    
                    # Store features in a list
                    feats = []
                    dis_out = model(dis.type(tensorType)[:, :, x:x+224, y:y+224])
                    feats_dis = feats

                    feats = []
                    ref_out = model(ref.type(tensorType)[:, :, x:x+224, y:y+224])
                    feats_ref = feats

                    # If not embedding then concat features
                    if "embed" not in args.model:
                        if "vitb32" in args.model:
                            feats_dis = torch.cat([feats_dis[i].detach().reshape([50,768]).unsqueeze(0) for i in range(12)])
                            feats_ref = torch.cat([feats_ref[i].detach().reshape([50,768]).unsqueeze(0) for i in range(12)])
                        
                            # batch processing for later...
                            feats_dis = feats_dis.unsqueeze(0)
                            feats_ref = feats_ref.unsqueeze(0)
                        elif "imagenet_vit" in args.model:
                            feats_dis = torch.cat([feats_dis[i][0] for i in range(len(feats_ref))])
                            feats_ref = torch.cat([feats_ref[i][0] for i in range(len(feats_ref))])
                            
                            # batch processing for later...
                            feats_dis = feats_dis.unsqueeze(0)
                            feats_ref = feats_ref.unsqueeze(0)
                        elif "dino" in args.model:
                            feats_dis = torch.cat([feats_dis[i][0] for i in range(len(feats_ref))])
                            feats_ref = torch.cat([feats_ref[i][0] for i in range(len(feats_ref))])
                            
                            # batch processing for later...
                            feats_dis = feats_dis.unsqueeze(0)
                            feats_ref = feats_ref.unsqueeze(0)
                        # skip for clip convnext and restnet versions


                    if args.distance == "l2":
                        if "convnext" in args.model or "rn50" in args.model:
                            score = np.sum([l2(feats_dis[i], feats_ref[i]) for i in range(len(feats_ref))])
                        else:
                            score = np.sum([l2(feats_dis[:,i], feats_ref[:,i]) for i in range(feats_ref.shape[1])])
                    elif args.distance == "swd":
                        score = np.sum(swd_dist(feats_dis, feats_ref, device))
                    elif args.distance == "cos":
                        if "embed" in args.model:
                            score = np.array(cos_dist(dis_out, ref_out))
                        elif "vitb32" in args.model or "dino" in args.model or "imagenet_vit" in args.model:
                            score = np.sum([cos_dist(feats_dis[:,i], feats_ref[:,i]) for i in range(feats_ref.shape[1])])
                        elif "convnext" in args.model:
                            score = np.sum([cos_dist(feats_dis[i], feats_ref[i]) for i in range(len(feats_ref))])
                        elif "rn50" in args.model:
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
            
        dis_name = data_i["dis_im"].split("/")[-1]
        ref_name = data_i["ref_im"].split("/")[-1]
        mos = float(data_i["mos"])
        f.write(dis_name+","+ref_name+","+str(mos)+","+str(d)+"\n")
    f.close()