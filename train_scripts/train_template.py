# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys

# option handler
import optionHandler
options = optionHandler.optionHandler(sys.argv)

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as df

options.initArguments()

# load samples
input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage(), options.getAddSampleSuffix())

weight_expr = 'x.scaleFactor_ELE * x.scaleFactor_MUON * x.scaleFactor_TRIGGER * x.scaleFactor_PILEUP * x.scaleFactor_ZVERTEX * x.mcWeight * x.scaleFactor_BTAG'

# define all samples
input_samples.addSample(options.getDefaultName("ttbar")  , label = "ttbar"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
input_samples.addSample(options.getDefaultName("singletop")  , label = "singletop"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
input_samples.addSample(options.getDefaultName("W")  , label = "W"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)
input_samples.addSample(options.getDefaultName("Z")  , label = "Z"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr)


# additional samples for adversary training
if options.isAdversary():
    input_samples.addSample(options.getAddSampleName("ttmb"), label = "ttmb"+options.getAddSampleSuffix(), normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
    input_samples.addSample(options.getAddSampleName("ttbb"), label = "ttbb"+options.getAddSampleSuffix(), normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
    input_samples.addSample(options.getAddSampleName("tt2b"), label = "tt2b"+options.getAddSampleSuffix(), normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
    input_samples.addSample(options.getAddSampleName("ttb") , label = "ttb"+options.getAddSampleSuffix() , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )

if options.isBinary():
    input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

if not options.isAdversary():
    # initializing DNN training class
    dnn = DNN.DNN(
        save_path       = options.getOutputDir(),
        input_samples   = input_samples,
        category_name   = options.getCategory(),
        train_variables = options.getTrainVariables(),
        # number of epochs
        train_epochs    = options.getTrainEpochs(),
        # metrics for evaluation (c.f. KERAS metrics)
        eval_metrics    = ["acc"],
        # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
        test_percentage = options.getTestPercentage(),
        # balance samples per epoch such that there amount of samples per category is roughly equal
        balanceSamples  = options.doBalanceSamples(),
        evenSel         = options.doEvenSelection(),
        norm_variables  = options.doNormVariables(),
        # implementation of options when using Domain Adaptation
        domain_adaptation = options.doDomainAdaptation(),
        grad_reversal_lambda = options.getGradReversalLambda()
        )
else:
    import DRACO_Frameworks.DNN.CAN as CAN
    # initializing CAN training class
    dnn = CAN.CAN(
        save_path       = options.getOutputDir(),
        input_samples   = input_samples,
        category_name   = options.getCategory(),
        train_variables = options.getTrainVariables(),
        # number of epochs
        train_epochs    = options.getTrainEpochs(),
        # metrics for evaluation (c.f. KERAS metrics)
        eval_metrics    = ["acc"],
        # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
        test_percentage = options.getTestPercentage(),
        # balance samples per epoch such that there amount of samples per category is roughly equal
        balanceSamples  = options.doBalanceSamples(),
        evenSel         = options.doEvenSelection(),
        norm_variables  = options.doNormVariables(),
        addSampleSuffix = options.getAddSampleSuffix())


# build DNN model
dnn.build_model(config=options.getNetConfig(), penalty=options.getPenalty())

# perform the training
dnn.train_model()

# evalute the trained model
dnn.eval_model()

# save information
dnn.save_model(sys.argv, filedir, options.getNetConfigName(), get_gradients = options.doGradients())

# save and print variable ranking according to the input layer weights
dnn.get_input_weights()

# save and print variable ranking according to all layer weights
dnn.get_weights()


# variation plots
if options.doVariations():
    dnn.get_variations(options.isBinary())

# plotting
if options.doPlots():
    # plot the evaluation metrics
    dnn.plot_metrics(privateWork = options.isPrivateWork())

    if options.isBinary():
        # plot output node
        bin_range = options.getBinaryBinRange()
        dnn.plot_binaryOutput(
            log         = options.doLogPlots(),
            privateWork = options.isPrivateWork(),
            printROC    = options.doPrintROC(),
            nbins       = 15,
            bin_range   = bin_range,
            name        = options.getName(),
            sigScale    = options.getSignalScale())
        if options.isAdversary():
            dnn.plot_ttbbKS_binary(
                log                 = options.doLogPlots(),
                signal_class        = options.getSignal(),
                privateWork         = options.isPrivateWork())
    else:
        # plot the output nodes
        if options.doDomainAdaptation():
            dnn.plot_domain_output()
            # dnn.plot_domain_outputNodes(  
            #     log                 = options.doLogPlots(),
            #     privateWork         = options.isPrivateWork(),
            #     printROC            = False,
            #     sigScale            = options.getSignalScale())

        # plot the confusion matrix
        dnn.plot_confusionMatrix(
            privateWork = options.isPrivateWork(),
            printROC    = options.doPrintROC())

        # plot the output nodes
        dnn.plot_outputNodes(
            log                 = options.doLogPlots(),
            privateWork         = options.isPrivateWork(),
            printROC            = options.doPrintROC(),
            sigScale            = options.getSignalScale())

        # plot the output discriminators
        dnn.plot_discriminators(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            printROC            = options.doPrintROC(),
            sigScale            = options.getSignalScale())

        # plot event yields
        dnn.plot_eventYields(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            sigScale            = options.getSignalScale())

        # plot closure test
        # dnn.plot_closureTest(
        #     log                 = options.doLogPlots(),
        #     signal_class        = options.getSignal(),
        #     privateWork         = options.isPrivateWork())

        # plot ttbb KS test
        if options.isAdversary():
            dnn.plot_ttbbKS(
                log                 = options.doLogPlots(),
                signal_class        = options.getSignal(),
                privateWork         = options.isPrivateWork())


if options.doGradients():
    dnn.get_gradients(options.isBinary())

if options.doDomainAdaptation():
    dnn.save_da_information()
