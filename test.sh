for i in {1..15}
do if [ $i -lt 10 ]
then
    git add "0$i"
fi 
if [ $i -ge 10 ]
then
    git add "$i"
fi
done