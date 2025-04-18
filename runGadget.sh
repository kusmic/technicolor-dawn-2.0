                                                                                                runGadget.sh                                                                                                               
# reset everything
rm -r output

# get latest from repository
git pull origin

# build
make clean
make -j4

# run Gadget
mpirun -np 4 ./Gadget4 param.txt | tee output.log

# make output animations
python3 plotOutputAnimation.py
python3 plotFeedbackMap.py
