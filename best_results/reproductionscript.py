# Usage: python reproductionscript.py -f iemocap_arousal.csv

import pandas as pd
import sys
import json

args = sys.argv[1:]
fIdx = args.index('-f')
filename = args[fIdx+1]

df = pd.read_csv(filename)

key = 'ccc'
corrFiles = {'mosi_sentiment.csv', 'mosei_sentiment.csv', 'mosei_happiness.csv'}
if filename in corrFiles:
    key = 'corr'
# bestParsed = df['best_metric_result'].map(lambda x : json.loads(x.replace("\'", "\""))[key])
# params = df.loc[bestParsed.idxmax()]

# Iterate through all the best ones since the CSV files only contain optimal ones
for i in range(len(df)):
    params = df.loc[i]
    
    exclude = {'save_checkpoint_filepath', 'best_metric_result', 'metric_list', 'load_checkpoint_filepath', 'loss_fn_name',
    'feature_input_dim', 'feature_hidden_dim', 'feature_output_dim', 'projection_output_dim', 
    'load_pretrain_contrastive', 'load_pretrain_residual'}
    store_true_params = {'multitask'}
    store_false_params = {'contrastive', 'in_batch_cl', 'freeze_contrastive_param'}
    s = 'python run_main.py '
    for k,v in params.items():
        if k not in exclude:
            if k in store_true_params: # default: False, flag means True
                if v == 'True':
                   s += f'--{k}'
                else:
                    pass # default False
            elif k in store_false_params: # default: True, flag means False
                if v == 'False':
                   s += f'--{k}'
                else:
                    pass # default True
            else:
                s += f'--{k} {v} '
    s += '--dataset_dir /work/siyuanwu/meta/ --output_dir output/'
    print(s)
