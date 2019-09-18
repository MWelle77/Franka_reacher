for i in {1..2};
do
    python main.py --random-task --num-frames 100000000 --num-processes 4 test_franka
done
