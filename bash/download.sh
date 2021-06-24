

START=1
for i in $(eval echo "{$START..$1}")
	do
	wget -O target/$i.jpeg https://picsum.photos/200
	done
