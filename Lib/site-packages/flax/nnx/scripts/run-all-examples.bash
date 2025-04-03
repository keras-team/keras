set -e

source .venv/bin/activate

for f in $(find examples/nnx_toy_examples -name "*.py" -maxdepth 1); do
    echo -e "\n---------------------------------"
    echo "$f"
    echo "---------------------------------"
    MPLBACKEND=Agg python "$f"
done
