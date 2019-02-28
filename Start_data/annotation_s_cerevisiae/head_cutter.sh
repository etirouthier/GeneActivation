#!/bin/bash

for num in `ls *.gff3`
do 
	tail -n +8 $num > fich.tmp 
	mv fich.tmp $num
done
