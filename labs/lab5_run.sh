for i in $(seq 2 10);
do
    mpiexec -n $i python lab5.py
done