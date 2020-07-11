echo ${2:-"output"}  ${1:-"../../data/amap_traffic_train/000004/2.jpg"}
python inference.py configs/culane.py --test_model checkpoints/culane_18.pth \
    --test_work_dir ${2:-"output"} \
    --test_img ${1:-"../../data/amap_traffic_train/000004/2.jpg"}
