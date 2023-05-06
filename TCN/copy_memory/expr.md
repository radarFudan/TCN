It seems the first three step the weighted loss is better, but the terminal results does not show the superiority

## vanilla

python3 copymem_test.py --power 0
python3 copymem_test.py --power 2

## Longer seq

python3 copymem_test.py --power 0 --seq_len 20 
python3 copymem_test.py --power 0 --seq_len 20 

## Add gradient evaluation

_g

## Repeat for checking loss-accuracy relationship

`python3 copymem_test.py --power 0 --epochs 500 > benchmark1_repeat4.txt`
`python3 copymem_test.py --power 2 --epochs 500 > WL1_repeat4.txt`

`python3 copymem_test.py --power 0 --seq_len 20 --epochs 500 > benchmark2_repeat4.txt`
`python3 copymem_test.py --power 2 --seq_len 20 --epochs 500 > WL2_repeat4.txt`

## 