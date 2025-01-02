HOURS=$(cat info.out | grep hours | awk '{sum += $4} END {print sum}')
MINUTES=$(cat info.out | grep minutes | awk '{sum += $4} END {print sum}')
SECONDS=$(cat info.out | grep seconds | awk '{sum += $4} END {print sum}')

echo $HOURS 'hours' $MINUTES 'minutes' $SECONDS 'seconds'
