#!/bin/bash

# If a config package is given, extract and use it
if [ -f /config.tar.gz ]; do
  tar -C /src/RnD.PointcloudClassification.Points3D -x -f /config.tar.gz
done

echo "Entering torch-points3d environment"
echo "Command given: $@"

set -e

/bin/bash --login -c "/root/miniconda3/bin/conda init bash"
#/bin/bash --login -c "/root/miniconda3/bin/conda activate $ENV_PREFIX" || true

exec "$@"
