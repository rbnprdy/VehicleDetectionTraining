export PIPELINE_CONFIG_PATH=/localhome/rubenpurdy/Developer/SeniorDesign/training/models/ssdlite_mobilenet_v2_coco_2018_05_09/ssdlite_mobilenet_v2_vehicle_detection.config
export MODEL_DIR=/localhome/rubenpurdy/Developer/SeniorDesign/training/models/ssdlite_mobilenet_v2_coco_2018_05_09/
export NUM_TRAIN_STEPS=50000
export SAMPLE_1_OF_N_EVAL_EXAMPLES=1
echo "cd /localhome/rubenpurdy/tensorflow/models/research &&
python3 object_detection/model_main.py \
    --pipeline_config_path=\${PIPELINE_CONFIG_PATH} \
    --model_dir=\${MODEL_DIR} \
    --num_train_steps=\${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=\$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr" | xclip
