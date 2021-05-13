#!/bin/bash
echo "Entering torch-points3d environment"
echo "Command given: $@"

set -e

/bin/bash --login -c "/root/miniconda3/bin/conda init bash"
#/bin/bash --login -c "/root/miniconda3/bin/conda activate $ENV_PREFIX" || true

exec "$@"