# DONE:
1. env set: same as streammapnet https://github.com/yuantianyuan01/StreamMapNet/tree/main
2. data: 
    Download NuScenes dataset to ./datasets/nuScenes.
    python tools/data_converter/StreamMap_nuscenes_converter.py --data-root ./datasets/nuScenes --newsplit
3. pretrain: 
bash tools/dist_train.sh ./IC_plugin/configs/track/nusc_newsplit_480_60x30_24e_tracker_asso.py num_gpus


# TODO:
1. stage2 train code;
2. metrics;
3. weights;
