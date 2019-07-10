GPU=1
CUDA_VISIBLE_DEVICES=${GPU} \
python model-test.py \
	--valroot reg_dataset/cute80_288 \
	--workers 2 \
	--batchSize 100 \
	--niter 10 \
	--lr 1 \
	--cuda \
	--displayInterval 100 \
	--valInterval 1000 \
	--saveInterval 40000 \
	--adadelta \
	--BidirDecoder
