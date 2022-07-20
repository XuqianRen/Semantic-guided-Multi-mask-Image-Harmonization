
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model ht --tr_r_enc_head 2 \
--tr_r_enc_layers 9 --name FCHT_2H9L_Multi_LAB --dataset_root path \
--dataset_name IHD --batch_size 8 --gpu_ids 0,1,2,3 --init_port 3202 

