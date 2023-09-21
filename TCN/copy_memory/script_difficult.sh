CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 20 --power -1 --epochs 50 > reproduce/WL2__1.txt
CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 20 --power 0 --epochs 50 > reproduce/WL2_0.txt
CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 20 --power 1 --epochs 50 > reproduce/WL2_1.txt
CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 20 --power 2 --epochs 50 > reproduce/WL2_2.txt
CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 20 --power 10 --epochs 50 > reproduce/WL2_10.txt
CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 20 --power 100 --epochs 50 > reproduce/WL2_inf.txt
