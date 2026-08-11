#!/bin/bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
export AWS_ENDPOINT_URL=http://localhost:9000
PYTHON=/home/mr_tarouzi/bigdata-cluster/big-data-project/.venv/bin/python

# echo "Running close_f1_chapter.py..."
# $PYTHON /home/mr_tarouzi/bigdata-cluster/big-data-project/analytics/close_f1_chapter.py \
#   --previous-path s3a://processed-data/dataset_eligibilite_final \
#   --candidate-path s3a://processed-data/dataset_eligibilite_test \
#   > /home/mr_tarouzi/bigdata-cluster/big-data-project/analytics/f1_bucket_comparison/run.log 2>&1

echo "Running ceiling_diagnostics.py..."
$PYTHON /home/mr_tarouzi/bigdata-cluster/big-data-project/analytics/ceiling_diagnostics.py \
  --data s3a://processed-data/dataset_eligibilite_test \
  --outdir /home/mr_tarouzi/bigdata-cluster/big-data-project/analytics/diagnostics_output \
  > /home/mr_tarouzi/bigdata-cluster/big-data-project/analytics/diagnostics_output/run.log 2>&1

echo "Running test_cleaned_dataset_battery.py..."
$PYTHON /home/mr_tarouzi/bigdata-cluster/big-data-project/analytics/test_cleaned_dataset_battery.py \
  --original_path s3a://processed-data/dataset_eligibilite \
  --clean_path s3a://processed-data/dataset_eligibilite_test \
  > /home/mr_tarouzi/bigdata-cluster/big-data-project/analytics/battery_run.log 2>&1

echo "DONE"
