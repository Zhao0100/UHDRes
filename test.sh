LOG_DIR="/home/jovyan/boomcheng-work-shcdt/zhaoshihao/code/UHD2/logs"
mkdir -p $LOG_DIR
CUDA_VISIBLE_DEVICES=0 python inference.py > $LOG_DIR/uhdres_UHD_LL_1031.log 2>&1 &
