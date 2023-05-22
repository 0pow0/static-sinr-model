Task: Predict static SINR (max)

Input: Features of attached base station (distance, Tx Power, Bandwidth, Subband Bandwidth. Subband offset)

Output: Static SINR in **dB**

Data: Included and located under data/folder. `.npy` data file could be generated with `data_gen.py` script.

Overwrite the `--train_folder` and `--val_folder` to the absolute path of data/train and data/test respectively; Overwrite the `--checkpoint_dir` to the absolute path of the folder to store model parameters.

Start training with:

```python
cat args.txt | xargs python main.py
```

Model parameters would be stored in that folder

Test with:
```python
python util/plot.py
```