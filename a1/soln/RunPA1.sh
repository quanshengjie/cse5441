#
# Batch submission script for Prog. Assignment 1
# If this script is placed in file RunPA1.bat,
# to submit batch job, use "qsub RunPA1.bat"
# Status of batch job can be checked using "qstat -u osu...." 
#
#
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=12
#PBS -N PA1
#PBS -S /bin/ksh
#PBS -j oe
cd $TMPDIR
#
# Assumes the source program is in file pa2-p1-sol.c in a
# subdirectory called 5441/Au14/PA1 in the OSC home directory
# Change it appropriately to match your directory and file name
#
cp $HOME/5441/a1/soln/pa1-p1-sol.c .
cp $HOME/5441/a1/soln/pa1-p2-sol.c .
cp $HOME/5441/a1/soln/pa1-p3-sol.c .

echo " "; echo "--------------------------------------------------------------"
echo "Compiling Programming Assignment 1, Problem 1"
echo "--------------------------------------------------------------"; echo " "
gcc -O3 -fopenmp -o pa1-p1-gcc pa1-p1-sol.c
icc -fast -openmp -o pa1-p1-icc pa1-p1-sol.c
echo " "; echo "ICC: Run 1"; echo " "
./pa1-p1-icc
echo " "; echo "GCC: Run 1"; echo " "
./pa1-p1-gcc
echo " "; echo "ICC: Run 2"; echo " "
./pa1-p1-icc
echo " "; echo "GCC: Run 2"; echo " "
./pa1-p1-gcc
echo " "; echo "ICC: Run 3"; echo " "
./pa1-p1-icc
echo " "; echo "GCC: Run 3"; echo " "
./pa1-p1-gcc


echo " "; echo "--------------------------------------------------------------"
echo "Compiling Problem 2"
echo "--------------------------------------------------------------"; echo " "
gcc -O3 -fopenmp -o pa1-p2-gcc pa1-p2-sol.c
icc -fast -openmp -o pa1-p2-icc pa1-p2-sol.c
echo " "; echo "ICC: Run 1"; echo " "
./pa1-p2-icc
echo " "; echo "GCC: Run 1"; echo " "
./pa1-p2-gcc
echo " "; echo "ICC: Run 2"; echo " "
./pa1-p2-icc
echo " "; echo "GCC: Run 2"; echo " "
./pa1-p2-gcc
echo " "; echo "ICC: Run 3"; echo " "
./pa1-p2-icc
echo " "; echo "GCC: Run 3"; echo " "
./pa1-p2-gcc


echo " "; echo "--------------------------------------------------------------"
echo "Compiling Problem 3"
echo "--------------------------------------------------------------"; echo " "
gcc -O3 -fopenmp -o pa1-p3-gcc pa1-p3-sol.c
icc -fast -openmp -o pa1-p3-icc pa1-p3-sol.c
echo " "; echo "ICC: Run 1"; echo " "
./pa1-p3-icc
echo " "; echo "GCC: Run 1"; echo " "
./pa1-p3-gcc
echo " "; echo "ICC: Run 2"; echo " "
./pa1-p3-icc
echo " "; echo "GCC: Run 2"; echo " "
./pa1-p3-gcc
echo " "; echo "ICC: Run 3"; echo " "
./pa1-p3-icc
echo " "; echo "GCC: Run 3"; echo " "
./pa1-p3-gcc

