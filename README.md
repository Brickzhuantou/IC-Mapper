# DONE:
1. env set: same as streammapnet https://github.com/yuantianyuan01/StreamMapNet/tree/main
2. data: 
    Download NuScenes dataset to ./datasets/nuScenes.
    python tools/data_converter/StreamMap_nuscenes_converter.py --data-root ./datasets/nuScenes --newsplit
3. pretrain: 
bash tools/dist_train.sh ./IC_plugin/configs/track/nusc_newsplit_480_60x30_24e_tracker_asso.py num_gpus
4. train（stage2）:
bash tools/dist_train.sh ./IC_plugin/configs/track_fusion/nusc_newsplit_480_60x30_24e_tracker_asso_fusion.py num_gpus

# TODO:
1. metrics;
2. weights;
