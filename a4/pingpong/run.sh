max=1048576
niters=100

echo -e "### BLOCKING ###"
for ((sz=1;sz<=$max;sz*=2))
do
    mpiexec -n 2 ./blocking  $sz $niters
done

echo -e "### BLOCKING ###"
for ((sz=1;sz<=$max;sz*=2))
do
    mpiexec -n 2 ./nonblocking  $sz $niters
done
