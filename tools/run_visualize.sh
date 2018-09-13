src=$1
dst=$2
cnd=$3

python visualize.py \
	--num_labels=100 \
	--scores=../analysis/predictions.csv \
	--truths=../analysis/truth.csv \
	--src="${src}" \
	--dst="${dst}" \
	--cnd="${cnd}"
