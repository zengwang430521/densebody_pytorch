source activate densebody
python train_gcn.py --cuda --ngpu 8 --workers 8 --max_dataset_size -1 --save_result_freq 500  --save_epoch_freq 2  --batch_size 32 --niter 20
