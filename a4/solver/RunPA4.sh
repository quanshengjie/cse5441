#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=12
#PBS -N PA4
#PBS -S /bin/ksh
#PBS -j oe

ITER=3

module load mvapich2
cd $HOME/cse5441/a4/solver

for np in 1 4 8 12
do
   for((i=1;i<=$ITER;i++))
   do
        echo "### nprocs=$np iteration=$i"
        mpiexec -n $np ./solver
	sleep 1
   done
done
