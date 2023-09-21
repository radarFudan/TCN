CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --power -1 --epochs 50 > reproduce/WL_1.txt
CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --power 0 --epochs 50 > reproduce/WL0.txt
CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --power 1 --epochs 50 > reproduce/WL1.txt
CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --power 2 --epochs 50 > reproduce/WL2.txt
CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --power 10 --epochs 50 > reproduce/WL10.txt
CUDA_VISIBLE_DEVICES=0 python3 copymem_test.py --power 100 --epochs 50 > reproduce/WLinf.txt
