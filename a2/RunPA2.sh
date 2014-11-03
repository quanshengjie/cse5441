#
# Batch submission script for Prog. Assignment 1
# If this script is placed in file RunPA1.bat,
# to submit batch job, use "qsub RunPA1.bat"
# Status of batch job can be checked using "qstat -u osu...." 
#
#
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=12
#PBS -N PA2
#PBS -S /bin/ksh
#PBS -j oe

ITER=2


echo " "; echo "--------------------------------------------------------------"
echo "Compiling Programming Assignment 2, Base version"
echo "--------------------------------------------------------------"; echo " "
gcc -O3 -lm -fopenmp -o 0_gcc 0.c
icc -fast -openmp -o 0_icc 0.c

for((i=1;i<=$ITER;i++))
do
	echo " "; echo "ICC: Threads: 1, Run $i"; echo " "
	./0_icc
	echo " "; echo "GCC: Threads: 1, Run $i"; echo " "
	./0_gcc
done


echo " "; echo "--------------------------------------------------------------"
echo "Compiling Problem 1"
echo "--------------------------------------------------------------"; echo " "
gcc -O3 -lm -fopenmp -o 1_gcc 1.c
icc -fast -openmp -o 1_icc 1.c
for((nt=2;nt<=8;nt*=2))
do
   for((i=1;i<=$ITER;i++))
   do
      echo " "; echo "ICC: Threads: $nt, Run $i"; echo " "
      ./1_icc $nt
      echo " "; echo "GCC: Threads: $nt, Run $i"; echo " "
      ./1_gcc $nt
   done
done

echo " "; echo "--------------------------------------------------------------"
echo "Compiling Problem 2"
echo "--------------------------------------------------------------"; echo " "
gcc -O3 -lm -fopenmp -o 2_gcc 2.c
icc -fast -openmp -o 2_icc 2.c
for((nt=2;nt<=8;nt*=2))
do
   for((i=1;i<=$ITER;i++))
   do
      echo " "; echo "ICC: Threads: $nt, Run $i"; echo " "
      ./2_icc $nt
      echo " "; echo "GCC: Threads: $nt, Run $i"; echo " "
      ./2_gcc $nt
   done
done

echo " "; echo "--------------------------------------------------------------"
echo "Compiling Problem 3"
echo "--------------------------------------------------------------"; echo " "
gcc -O3 -lm -fopenmp -o 3_gcc 3.c
icc -fast -openmp -o 3_icc 3.c
for((nt=2;nt<=8;nt*=2))
do
   for((i=1;i<=$ITER;i++))
   do
      echo " "; echo "ICC: Threads: $nt, Run $i"; echo " "
      ./3_icc $nt
      echo " "; echo "GCC: Threads: $nt, Run $i"; echo " "
      ./3_gcc $nt
   done
done

echo " "; echo "--------------------------------------------------------------"
echo "Compiling Problem 4"
echo "--------------------------------------------------------------"; echo " "
gcc -O3 -lm -fopenmp -o 4_gcc 4.c
icc -fast -openmp -o 4_icc 4.c
for((nt=2;nt<=8;nt*=2))
do
   for((i=1;i<=$ITER;i++))
   do
      echo " "; echo "ICC: Threads: $nt, Run $i"; echo " "
      ./4_icc $nt
      echo " "; echo "GCC: Threads: $nt, Run $i"; echo " "
      ./4_gcc $nt
   done
done


echo " "; echo "--------------------------------------------------------------"
echo "Compiling Problem 5"
echo "--------------------------------------------------------------"; echo " "
gcc -O3 -lm -fopenmp -o 5_gcc 5.c
icc -fast -openmp -o 5_icc 5.c
for((nt=2;nt<=8;nt*=2))
do
   for((i=1;i<=$ITER;i++))
   do
      echo " "; echo "ICC: Threads: $nt, Run $i"; echo " "
      ./5_icc $nt
      echo " "; echo "GCC: Threads: $nt, Run $i"; echo " "
      ./5_gcc $nt
   done
done


