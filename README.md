##### How to run the code

- Change `PATH_TO_DATASET_DIR` to your own path to datasets. For example, your data is in `/work/siyuanwu/meta/iemocap_arousal.csv`, then your `PATH_TO_DATASET_DIR` will be `/work/siyuanwu/meta/`

  ```bash
  python run_main.py \
  --dataset_name iemocap_arousal \
  --dataset_dir PATH_TO_DATASET_DIR \
  --feature_extractor_depth 4 \
  --projection_output_dim 32 \
  ... (other arguments)
  ```

- More arguments could be seen in `config/config.py`.





