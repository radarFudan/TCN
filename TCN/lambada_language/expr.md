totoal epochs = 100

Seems a bit too long...

batchsize from 20 to 32,
epochs from 100 to 30
add checkpoint

CUDA_VISIBLE_DEVICES=0 python3 lambada_test.py --batch_size 32 --epochs 30 --save model_benchmark.pt > benchmark.txt
CUDA_VISIBLE_DEVICES=1 python3 lambada_test.py --batch_size 32 --epochs 30 --save model_WL_benchmark.pt > WL_benchmark.txt