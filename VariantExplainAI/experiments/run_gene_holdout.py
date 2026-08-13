#!/usr/bin/env python
import subprocess,sys
cmd=[sys.executable,'experiments/train_main.py','--config','configs/config.yaml','--train','data/splits/gene_train.csv','--val','data/splits/gene_val.csv','--test','data/splits/gene_test.csv','--out','outputs/gene_holdout']
raise SystemExit(subprocess.call(cmd))
