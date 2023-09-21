CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 100 --power -1 --epochs 50 > reproduce/WL3__1.txt
CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 100 --power 0 --epochs 50 > reproduce/WL3_0.txt
CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 100 --power 1 --epochs 50 > reproduce/WL3_1.txt
CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 100 --power 2 --epochs 50 > reproduce/WL3_2.txt
CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 100 --power 10 --epochs 50 > reproduce/WL3_10.txt
CUDA_VISIBLE_DEVICES=1 python3 copymem_test.py --seq_len 100 --power 100 --epochs 50 > reproduce/WL3_inf.txt
