TRAIN_IMAGE_DIR=/localhome/rubenpurdy/cocoapi/images/train2017/
VAL_IMAGE_DIR=/localhome/rubenpurdy/cocoapi/images/val2017/
TEST_IMAGE_DIR=/localhome/rubenpurdy/cocoapi/images/test2017/
TRAIN_ANNOTATIONS_FILE=/localhome/rubenpurdy/cocoapi/annotations/instances_train2017.json
VAL_ANNOTATIONS_FILE=/localhome/rubenpurdy/cocoapi/annotations/instances_val2017.json
TESTDEV_ANNOTATIONS_FILE=/localhome/rubenpurdy/cocoapi/annotations/image_info_test-dev2017.json
OUTPUT_DIR=$1

echo "python3 create_coco_tf_record.py --logtostderr \
       --train_image_dir=${TRAIN_IMAGE_DIR} \
       --val_image_dir=${VAL_IMAGE_DIR} \
       --test_image_dir=${TEST_IMAGE_DIR} \
       --train_annotations_file=${TRAIN_ANNOTATIONS_FILE} \
       --val_annotations_file=${VAL_ANNOTATIONS_FILE} \
       --testdev_annotations_file=${TESTDEV_ANNOTATIONS_FILE} \
       --output_dir=${OUTPUT_DIR}" | xclip
