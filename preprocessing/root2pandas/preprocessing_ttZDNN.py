import os
import sys
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(os.path.dirname(filedir))
sys.path.append(basedir)

import root2pandas
import variable_sets.ttXRecoVariables as variable_set



# define a base event selection which is applied for all Samples
base_selection = "\
( \
(N_Jets >= 4 and N_BTagsM >= 3 and Evt_Pt_MET > 20. and Weight_GEN_nom > 0. and genTTX_M_TopLep > 0.) \
and (\
(N_LooseMuons == 0 and N_TightElectrons == 1 and (Triggered_HLT_Ele35_WPTight_Gsf_vX == 1 or Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX == 1)) \
or \
(N_LooseElectrons == 0 and N_TightMuons == 1 and Muon_Pt > 29. and Triggered_HLT_IsoMu27_vX == 1) \
) \
)"


# define other additional selections
#ttbar_selection = "(\
#abs(Weight_scale_variation_muR_0p5_muF_0p5) <= 100 and \
#abs(Weight_scale_variation_muR_0p5_muF_1p0) <= 100 and \
#abs(Weight_scale_variation_muR_0p5_muF_2p0) <= 100 and \
#abs(Weight_scale_variation_muR_1p0_muF_0p5) <= 100 and \
#abs(Weight_scale_variation_muR_1p0_muF_1p0) <= 100 and \
#abs(Weight_scale_variation_muR_1p0_muF_2p0) <= 100 and \
#abs(Weight_scale_variation_muR_2p0_muF_0p5) <= 100 and \
#abs(Weight_scale_variation_muR_2p0_muF_1p0) <= 100 and \
#abs(Weight_scale_variation_muR_2p0_muF_2p0) <= 100 \
#)"

ttbar_selection = None
ttH_selection = "(Evt_Odd == 1)"
ttZ_selection = None

# define output classes
ttH_categories = root2pandas.EventCategories()
ttH_categories.addCategory("ttHbb", selection = None)

ttZ_categories = root2pandas.EventCategories()
ttZ_categories.addCategory("ttZbb", selection = None)

ttZJets_categories = root2pandas.EventCategories()
ttZJets_categories.addCategory("ttZJets", selection = None)

ttbar_categories = root2pandas.EventCategories()
ttbar_categories.addCategory("ttbb", selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("tt2b", selection = "(GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttb",  selection = "(GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttlf", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttcc", selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")


# initialize dataset class
dataset = root2pandas.Dataset(
    outputdir   = "/nfs/dust/cms/user/vdlinden/DNNInputFiles/ttZ_DNN_v3/",
    naming      = "dnn",
    addCNNmap   = False,
    addMEM      = False)

# add base event selection
dataset.addBaseSelection(base_selection)



ntuplesPath = "/nfs/dust/cms/user/vdlinden/ntuples2019/ttZ_DNN_v1/"

# add samples to dataset
dataset.addSample(
    sampleName  = "ttHbb",
    ntuples     = ntuplesPath+"/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    categories  = ttH_categories,
    selections  = ttH_selection)
    
dataset.addSample(
    sampleName  = "ttHNobb",
    ntuples     = ntuplesPath+"/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    categories  = ttH_categories,
    selections  = ttH_selection)

dataset.addSample(
    sampleName  = "ttZqq",
    ntuples     = ntuplesPath+"/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8_v2/*nominal*.root",
    categories  = ttZ_categories,
    selections  = ttZ_selection)

dataset.addSample(
    sampleName  = "ttZll",
    ntuples     = ntuplesPath+"/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/*nominal*.root",
    categories  = ttZ_categories,
    selections  = ttZ_selection)

dataset.addSample(
    sampleName  = "TTToSL",
    ntuples     = ntuplesPath+"/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    categories  = ttbar_categories,
    selections  = ttbar_selection)

dataset.addSample(
    sampleName  = "TTToHad",
    ntuples     = ntuplesPath+"/TTToHadronic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    categories  = ttbar_categories,
    selections  = ttbar_selection)

dataset.addSample(
    sampleName  = "TTToLep",
    ntuples     = ntuplesPath+"/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    categories  = ttbar_categories,
    selections  = ttbar_selection)

dataset.addSample(
    sampleName  = "ttZJets",
    ntuples     = ntuplesPath+"/ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8/*nominal*.root",
    categories  = ttZJets_categories,
    selections  = ttZ_selection)

# initialize variable list 
dataset.addVariables(variable_set.all_variables)

# define an additional variable list
additional_variables = [
    "N_Jets",
    "N_BTagsM",
    #"GenAdd_BB_inacceptance_part",
    #"GenAdd_B_inacceptance_part",
    #"GenHiggs_BB_inacceptance_part",
    #"GenHiggs_B_inacceptance_part",
    #"GenTopHad_B_inacceptance_part",
    #"GenTopHad_QQ_inacceptance_part",
    #"GenTopHad_Q_inacceptance_part",
    #"GenTopLep_B_inacceptance_part",
    #"GenAdd_BB_inacceptance_jet",
    #"GenAdd_B_inacceptance_jet",
    #"GenHiggs_BB_inacceptance_jet",
    #"GenHiggs_B_inacceptance_jet",
    #"GenTopHad_B_inacceptance_jet",
    #"GenTopHad_QQ_inacceptance_jet",
    #"GenTopHad_Q_inacceptance_jet",
    #"GenTopLep_B_inacceptance_jet",
    "Weight_XS",
    "Weight_CSV",
    "Weight_GEN_nom",
    "Evt_ID", 
    "Evt_Run", 
    "Evt_Lumi"]

# add these variables to the variable list
dataset.addVariables(additional_variables)

# run the preprocessing
dataset.runPreprocessing()
