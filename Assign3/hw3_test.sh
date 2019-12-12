wget -O ./model_best.pth "https://www.dropbox.com/s/8nu19nofqzri7gy/model_best.pth?dl=1"
CUDA_VISIBLE_DEVICES=0 python3 Test.py -d $1 -o $2 -m model_best.pth