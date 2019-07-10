GPU=1
CUDA_VISIBLE_DEVICES=${GPU} \
python train.py \
	--train_nips reg_dataset/NIPS2014 \
	--train_cvpr reg_dataset/CVPR2016 \
	--valroot reg_dataset/cute80_288 \
	--workers 2 \
	--batchSize 64 \
	--niter 10 \
	--lr 1 \
	--cuda \
	--experiment output/ \
	--displayInterval 100 \
	--valInterval 1000 \
	--saveInterval 40000 \
	--adadelta \
	--BidirDecoder
