#!/bin/bash

HOMEPATH="$(pwd)/imageURL"
DSTPATH="$(pwd)/images"

FILELIST="$(ls $HOMEPATH)"

for i in $FILELIST
do
	DIRNAME=$(echo $i | cut -d. -f1)
	DIRPATH=$DSTPATH/"$DIRNAME"
	if [ ! -d "$DIRPATH" ]
	then
		mkdir $DIRPATH
		# echo $DIRNAME
	fi
	cnt=0
	while read line
	do
		wget -O $DIRPATH/$cnt.jpg $line -t 1
		# wget -P $DIRPATH $line
		if [ "$?" != 0 ]
		then
			# echo "//////////////////////"
			# echo $DIRPATH/$cnt.jpg
			rm $DIRPATH/$cnt.jpg
		else
			# echo "=========================="
			HEADER=$(xxd $DIRPATH/$cnt.jpg | head -1)
			HEADER=$(echo $HEADER | sed -n "s/.*0000000: \([0-9A-Za-z]*\).*/\1/p")
			echo $HEADER
			if [ $HEADER != ffd8 ]
			then
				# echo "////////NoNoNo!!"
				rm $DIRPATH/$cnt.jpg
			else
				# echo "jpg!!!!!!!!!!!"
				cnt=$((cnt+1))
			fi
		# 	echo "/////////////////////////// :: "$DIRPATH/$cnt.jpg
		fi
	done < $HOMEPATH/"$i"
	# echo $DIRPATH/$i
	# while read line
	# do
	# done < 
done