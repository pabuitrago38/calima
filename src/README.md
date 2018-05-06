To make various data plots.

```
# With all points.
python dataDescriptors.py \
  --mode classify_0sec \
  --out_plot_dir /Users/paola/Google\ Drive/000-Development/Calima/Graphs/rawData1-plots-all

# No zero points.
python dataDescriptors.py \
  --mode classify_5min \
  --out_plot_dir /Users/paola/Google\ Drive/000-Development/Calima/Graphs/rawData1-plots-no-zero
```


To train classification Time > 0 sec.

```
# Train.
python nn_train.py \
  --checkpoint_dir /tmp/calima_checkpoints/classification_0sec

epoch=80
logging_dir=/tmp/calima_checkpoints/classification_0sec

# Evaluate on training data.
# --> To plot the ROC
python roc.py \
  --checkpoint_path ${logging_dir}/epoch_${epoch}.pth \
  --out_plot_path ${logging_dir}/roc_epoch_${epoch}_train.png \
  --mode classify_0sec

# Evaluate on testing data.
# --> To plot the ROC
python roc.py \
  --in_data_file /Users/paola/Google\ Drive/000-Development/Calima/Data/rawData2-P-filled.txt \
  --ref_data_file /Users/paola/Google\ Drive/000-Development/Calima/Data/rawData1-P-filled.txt \
  --checkpoint_path ${logging_dir}/epoch_${epoch}.pth \
  --out_plot_path ${logging_dir}/roc_epoch_${epoch}_test.png \
  --mode classify_0sec
```


To train classification Time > 5 min (points with Time == 0 are removed.)

```
# Train.
python nn_train.py \
  --checkpoint_dir /tmp/calima_checkpoints/classification_5min

epoch=160

# Evaluate on training data.
python roc.py \
  --checkpoint_path /tmp/calima_checkpoints/classification_5min/epoch_${epoch}.pth \
  --out_plot_path /tmp/calima_checkpoints/classification_5min/roc_epoch_${epoch}_train.png \
  --mode classify_5min

# Evaluate on testing data.
python roc.py \
  --in_data_file /Users/paola/Google\ Drive/000-Development/Calima/Data/rawData2-P-filled.txt \
  --ref_data_file /Users/paola/Google\ Drive/000-Development/Calima/Data/rawData1-P-filled.txt \
  --checkpoint_path /tmp/calima_checkpoints/classification_5min/epoch_${epoch}.pth \
  --out_plot_path /tmp/calima_checkpoints/classification_5min/roc_epoch_${epoch}_test.png \
  --mode classify_5min
```


Regression

# Results
- Evaluated in the real solution space.
/private/tmp/calima_checkpoints/regression_test_evalReal
