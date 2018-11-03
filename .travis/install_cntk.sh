set -e
pip install cntk

# open mpi is needed for cntk
rm -rf ~/mpi
mkdir ~/mpi
pushd ~/mpi
wget http://cntk.ai/PythonWheel/ForKeras/depends/openmpi_1.10-3.zip
unzip ./openmpi_1.10-3.zip
sudo dpkg -i openmpi_1.10-3.deb
popd
