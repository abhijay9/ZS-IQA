import pandas as pd
from scipy import stats
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dat", '--dataset', type=str, default=['live', 'tid2013', 'pipal'], help='datasets to test on: [live],[tid2013],[pipal]')
    parser.add_argument("-f", '--scoreFilename', type=str, default="", help='score filename')
    parser.add_argument("-t", '--type', type=str, default='all', help='')
    args = parser.parse_args()

    if args.scoreFilename == "":
        ValueError
    else:
        # results = pd.read_csv("results/"+args.dataset+"_"+args.scoreFilename+".csv")
        # results = pd.read_csv("results_aug24/"+args.dataset+"_"+args.scoreFilename+".csv")
        # results = pd.read_csv("results_aug27/"+args.dataset+"_"+args.scoreFilename+".csv")
        # results = pd.read_csv("results_aug30/"+args.dataset+"_"+args.scoreFilename+".csv")
        results = pd.read_csv("results_downsampled/"+args.dataset+"_"+args.scoreFilename+".csv")
        if "pipal" in args.dataset:
            results["pred"] = -results["pred"]
        if "psnr" in args.scoreFilename:
            results["pred"] = -results["pred"]


    if args.type == "all":
        srcc = stats.spearmanr(list(results["pred"]), list(results["mos"]))[0]
        krcc = stats.kendalltau(list(results["pred"]), list(results["mos"]))[0]
        plcc = stats.pearsonr(list(results["pred"]), list(results["mos"]))[0]
    if args.type == "byref":
        srcc, krcc, plcc = [], [], []
        for ref in list(results["ref"].unique()):

            res = results[results["ref"]==ref]
            
            s = stats.spearmanr(list(res["pred"]), list(res["mos"]))
            k = stats.kendalltau(list(res["pred"]), list(res["mos"]))
            p = stats.pearsonr(list(res["pred"]), list(res["mos"]))

            srcc.append(s[0])
            krcc.append(k[0])
            plcc.append(p[0])
        srcc, krcc, plcc = np.mean(srcc), np.mean(krcc), np.mean(plcc),

    # print("srcc = "+str(srcc))
    # print("krcc = "+str(krcc))
    # print("plcc = "+str(plcc))
    print(args.scoreFilename+","+str(plcc)+","+str(srcc)+","+str(krcc))