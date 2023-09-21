CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --seq_len 150 --power -1 --epochs 50 > reproduce/WL4__1.txt
CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --seq_len 150 --power 0 --epochs 50 > reproduce/WL4_0.txt
CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --seq_len 150 --power 1 --epochs 50 > reproduce/WL4_1.txt
CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --seq_len 150 --power 2 --epochs 50 > reproduce/WL4_2.txt
CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --seq_len 150 --power 10 --epochs 50 > reproduce/WL4_10.txt
CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --seq_len 150 --power 100 --epochs 50 > reproduce/WL4_inf.txt
