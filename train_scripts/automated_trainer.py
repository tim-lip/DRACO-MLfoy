import os 
import sys
import optparse

# parse run number
parser = optparse.OptionParser(usage="")
parser.add_option("-n", "--runNumber", dest="run_number", default=1, type=int, metavar="RUNNUMBER")
args = parser.parse_args()
run_number = args[0].run_number
print("RUN NUMBER: "+str(run_number))

# lambdas to iterate through
lambdas = [0,2,4,6,8,10,20,30,40,50,60,70,80,90,100]

# activate Draco
for lambda_ in lambdas:
	print("######################## DOING LAMBDA="+str(lambda_)) 
	os.system("python train_template.py \
		-i /work/tlippmann/root2pandas/outputFiles \
		-o testrun_no"+str(run_number)+"/da_training"+"_lambda="+str(lambda_)+" \
		-v variables_ttbar_AtlasOpenData2012 \
		-n ttbar_AtlasOpenData2012_multiclassing \
		-c ge2j_ge1t \
		-e 1000 \
		--plot --printroc --da\
		--lambda "+str(lambda_))