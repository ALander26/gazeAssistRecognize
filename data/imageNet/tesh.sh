# TMP="IP=111.111.111.1 ID=bbb, MSG=3849"

# echo $TMP | sed -n "s/.*ID=\([0-9A-Za-z,]*\).*/\1/p" | sed -n "s/,.*/ /p"

# TMP="0000000: ffd8 ffe0 0010 4a46 4946 0001 0101 0048 ......JFIF.....H"

# # echo $TMP | sed -n "s/.*\([0-9A-Za-z,]*\).*/\1/p"
# echo $TMP | sed -n "s/.*0000000: \([0-9A-Za-z ]*\).*/\1/p"

TMP="HELLO WORLD"
INDICATOR="HELLO WORLD"

if [ $TMP = $INDICATOR ]
then
	echo "true"
else
	echo "false"
fi