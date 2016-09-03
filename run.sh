

for l in libs/* 
do 
	CP=$CP:$l 
done 

java -XX:MaxHeapSize=6g -cp $CP com.rc.Remix $@
