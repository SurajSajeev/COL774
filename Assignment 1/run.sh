#!/bin/sh

 
if [ $1 = 1 ];
then
python linearreg.py $2 $3 $4 $5        
elif [ $1 = 2 ];
then
python weightedlin.py $2 $3 $4 
elif [ $1 = 3 ];
then
python logistic.py $2 $3
else
python gaussian.py $2 $3 $4
fi
#python gaussian.py q4x.dat q4y.dat 1

