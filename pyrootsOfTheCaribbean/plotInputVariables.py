import ROOT
import os
import sys
import ROOT
import optparse
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

from evaluationScripts.plotVariables import variablePlotter

usage="usage=%prog [options] \n"
usage+="USE: python train_template.py -o DIR -v FILE -n STR -c STR -e INT -s INT -p -l --privatework --netconfig=STR --signalclass=STR --printroc "

parser = optparse.OptionParser(usage=usage)

parser.add_option("-o", "--outputdirectory", dest="outputDir",default="plots_test_training",
        help="DIR for output", metavar="outputDir")

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="InputFeatures",
        help="DIR for input", metavar="inputDir")

parser.add_option("-v", "--variableselection", dest="variableSelection",default="example_variables",
        help="FILE for variables used to train DNNs", metavar="variableSelection")

parser.add_option("-l", "--log", dest="log", action = "store_true", default=False,
        help="activate for logarithmic plots", metavar="log")

parser.add_option("-p", "--privatework", dest="privateWork", action = "store_true", default=False,
        help="activate Private Work option", metavar="privateWork")

parser.add_option("-r", "--ratio", dest="ratio", action = "store_false", default=True,
        help="deactivate ratio plot", metavar="ratio")

parser.add_option("--title", dest="ratioTitle", default="#frac{ttH}{ttbar}",
        help="STR #frac{PROCESS}{PROCESS}", metavar="log")

parser.add_option("-k", "--ksscore", dest="KSscore", action = "store_false", default=True,
        help="deactivate KSscore", metavar="KSscore")

parser.add_option("-s", "--scalesignal", dest="scaleSignal", default=-1,
        help="-1 to scale Signal to background Integral, FLOAT to scale Signal with float value, False to not scale Signal",
        metavar="scaleSignal")

parser.add_option("--lumiscale", dest="lumiScale", default=1,
        help="FLOAT to scale Luminosity", metavar="lumiScale")

(options, args) = parser.parse_args()

#import Variable Selection
if not os.path.isabs(options.variableSelection):
    sys.path.append(basedir+"/variable_sets/")
    variable_set = __import__(options.variableSelection)
elif os.path.exists(options.variableSelection):
    variable_set = __import__(options.variableSelection)
else:
    sys.exit("ERROR: Variable Selection File does not exist!")

#get input directory path
if not os.path.isabs(options.inputDir):
    data_dir = basedir+"/workdir/"+options.inputDir
elif os.path.exists(options.inputDir):
    data_dir=options.inputDir
else:
    sys.exit("ERROR: Input Directory does not exist!")

#get output directory path
if not os.path.isabs(options.outputDir):
    plot_dir = basedir+"/workdir/"+options.outputDir
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
else: 
    plot_dir=options.outputDir
    if not os.path.exists(options.outputDir):
        os.makedirs(plot_dir)
   

# plotting options
plotOptions = {
    "ratio":        options.ratio,
    "ratioTitle":   options.ratioTitle,
    "logscale":     options.log,
    "scaleSignal":  options.scaleSignal,
    "lumiScale":    float(options.lumiScale),
    "KSscore":      options.KSscore,
    "privateWork":  options.privateWork,
    }
#   scaleSignal:
#   -1:     scale to background Integral
#   float:  scale with float value
#   False:  dont scale

# additional variables to plot
additional_variables = [
    ]


# initialize plotter
plotter = variablePlotter(
    output_dir      = plot_dir,
    variable_set    = variable_set,
    add_vars        = additional_variables,
    plotOptions     = plotOptions
    )

# add samples
plotter.addSample(
    sampleName      = "ttH",
    sampleFile      = data_dir+"/ttHbb_dnn.h5",
    plotColor       = ROOT.kOrange,
    signalSample    = True)


plotter.addSample(
    sampleName      = "ttbb",
    sampleFile      = data_dir+"/ttbb_dnn.h5",
    plotColor       = ROOT.kRed+3)

plotter.addSample(
    sampleName      = "tt2b",
    sampleFile      = data_dir+"/tt2b_dnn.h5",
    plotColor       = ROOT.kRed+2)

plotter.addSample(
    sampleName      = "ttb",
    sampleFile      = data_dir+"/ttb_dnn.h5",
    plotColor       = ROOT.kRed-2)

plotter.addSample(
    sampleName      = "ttcc",
    sampleFile      = data_dir+"/ttcc_dnn.h5",
    plotColor       = ROOT.kRed+1)

plotter.addSample(
    sampleName      = "ttlf",
    sampleFile      = data_dir+"/ttlf_dnn.h5")



# add JT categories
plotter.addCategory("4j_ge3t")
#plotter.addCategory("5j_ge3t")
#plotter.addCategory("ge6j_ge3t")
#plotter.addCategory("ge4j_ge3t")


# perform plotting routine
plotter.plot(saveKSValues = options.KSscore)
