wget "https://www.dropbox.com/s/03qqoaiono128da/model_best.pth.tar1?dl=1"
wget "https://www.dropbox.com/s/w4dxolquone8qgp/model_best.pth.tar2?dl=1"
wget "https://www.dropbox.com/s/57w3m8k488thn9e/model_best.pth.tar3?dl=1"
wget "https://www.dropbox.com/s/abp52frlzqclzm3/model_best.pth.tar4?dl=1"
wget "https://www.dropbox.com/s/4vl0i98wv4uixow/model_best.pth.tar5?dl=1"
wget "https://www.dropbox.com/s/798ctab5x9bdhcp/model_best.pth.tar6?dl=1"
wget "https://www.dropbox.com/s/ikc1eootskbgl84/model_best.pth.tar7?dl=1"

python3 Test.py -t $1 -o prediction1.csv -m model_best.pth.tar1\?dl\=1
python3 Test.py -t $1 -o prediction2.csv -m model_best.pth.tar2\?dl\=1
python3 Test.py -t $1 -o prediction3.csv -m model_best.pth.tar3\?dl\=1
python3 Test.py -t $1 -o prediction4.csv -m model_best.pth.tar4\?dl\=1
python3 Test.py -t $1 -o prediction5.csv -m model_best.pth.tar5\?dl\=1
python3 Test.py -t $1 -o prediction6.csv -m model_best.pth.tar6\?dl\=1
python3 Test.py -t $1 -o prediction7.csv -m model_best.pth.tar7\?dl\=1
python3 Ensemble.py -o $2
