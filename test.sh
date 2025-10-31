LOG_DIR="~"
mkdir -p $LOG_DIR
CUDA_VISIBLE_DEVICES=0 python inference.py > $LOG_DIR/output.log 2>&1 &
