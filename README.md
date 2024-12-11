# ZS-IQA

### Run evaluation

```
python run.py -dat <dataset(pipal)> -m <method_name(clip_vitb32,dinov1,embed_clip_vitb32,embed_dinov1)> -d <distance(l2,cos,wsd,jsd,skld)> -s <saveas>
```

for robustness tests:

```
python run.py -dat <dataset(pipal)> -m <method_name(clip_vitb32,dinov1,embed_clip_vitb32,embed_dinov1)> -d <distance(l2,cos,wsd,jsd,skld)> -s <saveas> -rob <tra,sca,rot> -pct <pctPixels>
```

