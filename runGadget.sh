# reset everything
rm -r output

# run Gadget
mpirun -np 4 ./Gadget4 param.txt | tee output.log
