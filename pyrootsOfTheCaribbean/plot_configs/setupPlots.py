import ROOT
ROOT.gROOT.SetBatch(True)
import re
import numpy as np
import pandas as pd
import os

# dictionary for colors
def GetPlotColor( cls ):
    color_dict = {
        "ttZ":          ROOT.kCyan,
        "ttH":          ROOT.kRed+1,
        "ttHbb":        ROOT.kRed+1,
        "ttHnonbb":     ROOT.kYellow-7,
        "ttlf":         ROOT.kRed-7,
        "ttcc":         ROOT.kRed+1,
        "ttbb":         ROOT.kRed+3,
        "tt2b":         ROOT.kRed+2,
        "ttb":          ROOT.kRed-2,
        "tthf":         ROOT.kRed-3,
        "ttbar":        ROOT.kOrange,
        "ttmb":         ROOT.kRed-1,
        "ttb_bb":       ROOT.kRed-1,
        "ST":           ROOT.kRed-8,
        "tHq":          ROOT.kBlue+6,
        "tHW":          ROOT.kBlue+3,
        "sig":          ROOT.kCyan,
        "bkg":          ROOT.kOrange,

        "ttH_STXS_0":          ROOT.kAzure,
        "ttH_STXS_1":          ROOT.kBlue-1,
        "ttH_STXS_2":          ROOT.kBlue-2,
        "ttH_STXS_3":          ROOT.kBlue-3,
        "ttH_STXS_4":          ROOT.kBlue-4,

        "ttHbb_STXS_0":          ROOT.kAzure,
        "ttHbb_STXS_1":          ROOT.kBlue-1,
        "ttHbb_STXS_2":          ROOT.kBlue-2,
        "ttHbb_STXS_3":          ROOT.kBlue-3,
        "ttHbb_STXS_4":          ROOT.kBlue-4,

        'singletop':        ROOT.kBlue,
        'Z':                ROOT.kRed,
        'W':                ROOT.kGreen,

        'realData':         ROOT.kRed,
        'simulatedData':    ROOT.kGreen        
        }

    return color_dict[cls]

def GetyTitle(privateWork = False):
    # if privateWork flag is enabled, normalize plots to unit area
    if privateWork:
        return "normalized to unit area"
    # return "Events expected"
    return ""


# ===============================================
# SETUP OF HISTOGRAMS
# ===============================================
def setupHistogram(
        values,
        nbins, bin_range,
        xtitle, ytitle,
        color = ROOT.kBlack, filled = True, weights=None):
    # define histogram
    histogram = ROOT.TH1D(xtitle, "", nbins, *bin_range)
    histogram.Sumw2(True)
    if weights:
        for v, w in zip(values, weights):
            histogram.Fill(v, w)
    else:
        for v in values:
            histogram.Fill(v)

    histogram.SetStats(False)
    histogram.GetXaxis().SetTitle(xtitle)
    histogram.GetYaxis().SetTitle(ytitle)

    histogram.GetYaxis().SetTitleOffset(1.4)
    histogram.GetXaxis().SetTitleOffset(1.2)
    histogram.GetYaxis().SetTitleSize(0.055)
    histogram.GetXaxis().SetTitleSize(0.055)
    histogram.GetYaxis().SetLabelSize(0.055)
    histogram.GetXaxis().SetLabelSize(0.055)
    # histogram.GetYaxis().SetMaxDigits(2)

    histogram.SetMarkerColor(color)

    if filled:
        histogram.SetLineColor( ROOT.kBlack )
        histogram.SetFillColor( color )
        histogram.SetLineWidth(1)
    else:
        histogram.SetLineColor( color )
        histogram.SetFillColor(0)
        histogram.SetLineWidth(2)

    return histogram

def setupYieldHistogram(yields, classes, n_classes, xtitle, ytitle, color = ROOT.kBlack, filled = True):
    # define histogram
    histogram = ROOT.TH1D(xtitle.replace(" ","_"), "", n_classes, 0, n_classes)
    histogram.Sumw2(True)

    for iBin in range(len(classes)):
        if iBin>=n_classes: continue
        histogram.SetBinContent(iBin+1, yields[iBin])
        histogram.SetBinError(iBin+1, np.sqrt(yields[iBin]))


    histogram.SetStats(False)
    histogram.GetXaxis().SetTitle(xtitle)
    # histogram.GetYaxis().SetTitle(ytitle)

    histogram.GetYaxis().SetTitleOffset(1.4)
    histogram.GetXaxis().SetTitleOffset(1.2)
    histogram.GetYaxis().SetTitleSize(0.055)
    histogram.GetXaxis().SetTitleSize(0.055)
    histogram.GetYaxis().SetLabelSize(0.055)
    histogram.GetXaxis().SetLabelSize(0.055)

    histogram.SetMarkerColor(color)

    if filled:
        histogram.SetLineColor( ROOT.kBlack )
        histogram.SetFillColor( color )
        histogram.SetLineWidth(1)
    else:
        histogram.SetLineColor( color )
        histogram.SetFillColor(0)
        histogram.SetLineWidth(2)

    for i, cls in enumerate(["t#bar{t}", "t", "W", "Z"]):
        if i>=n_classes: continue
        histogram.GetXaxis().SetBinLabel(i+1, cls)

    return histogram


def setupConfusionMatrix(matrix, ncls, xtitle, ytitle, binlabel, errors = None):
    # check if errors for matrix are given
    has_errors = isinstance(errors, np.ndarray)
    #print(has_errors)

    # init histogram
    cm = ROOT.TH2D("confusionMatrix", "", ncls, 0, ncls, ncls, 0, ncls)
    cm.SetStats(False)
    ROOT.gStyle.SetPaintTextFormat(".3f")


    for xit in range(cm.GetNbinsX()):
        for yit in range(cm.GetNbinsY()):
            cm.SetBinContent(xit+1,yit+1, matrix[xit, yit])
            if has_errors:
                cm.SetBinError(xit+1,yit+1, errors[xit, yit])

    cm.GetXaxis().SetTitle("vorhergesagte Klasse")
    cm.GetYaxis().SetTitle("wahre Klasse")

    cm.SetMarkerColor(ROOT.kWhite)

    minimum = np.min(matrix)
    maximum = np.max(matrix)

    cm.GetZaxis().SetRangeUser(minimum, maximum)

    for xit in range(ncls):
        cm.GetXaxis().SetBinLabel(xit+1, binlabel[xit])
    for yit in range(ncls):
        cm.GetYaxis().SetBinLabel(yit+1, binlabel[yit])

    cm.GetXaxis().SetLabelSize(0.05)
    cm.GetYaxis().SetLabelSize(0.05)
    cm.SetMarkerSize(2.)
    if cm.GetNbinsX()>6:
        cm.SetMarkerSize(1.5)
    if cm.GetNbinsX()>8:
        cm.SetMarkerSize(1.)

    return cm



# ===============================================
# DRAW HISTOGRAMS ON CANVAS
# ===============================================
def drawConfusionMatrixOnCanvas(matrix, canvasName, catLabel, ROC = None, ROCerr = None, privateWork = False):
    # init canvas
    canvas = ROOT.TCanvas(canvasName, canvasName, 1024, 1024)
    canvas.SetTopMargin(0.15)
    canvas.SetBottomMargin(0.15)
    canvas.SetRightMargin(0.15)
    canvas.SetLeftMargin(0.15)
    canvas.SetTicks(1,1)

    # draw histogram
    #ROOT.gStyle.SetPalette(69)
    draw_option = "colz text1"
    if ROCerr: draw_option += "e"
    matrix.DrawCopy(draw_option)

    # setup TLatex
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)
    latex.SetTextSize(0.03)

    l = canvas.GetLeftMargin()
    t = canvas.GetTopMargin()

    # add category label
    # latex.DrawLatex(l,1.-t+0.01, catLabel)

    if privateWork:
        latex.DrawLatex(l, 1.-t+0.04, "CMS private work")

    # add ROC score if activated
    if ROC:
        text = "ROC-AUC = {:.3f}".format(ROC)
        if ROCerr:
            text += "#pm {:.3f}".format(ROCerr)
            latex.DrawLatex(l+0.4,1.-t+0.01, text)
        else:
            latex.DrawLatex(l+0.47,1.-t+0.01, text)

    return canvas


def drawClosureTestOnCanvas(sig_train, bkg_train, sig_test, bkg_test, plotOptions, canvasName):
    canvas = getCanvas(canvasName)

    # move over/underflow bins into plotrange
    moveOverUnderFlow(sig_train)
    if not bkg_train is None: moveOverUnderFlow(bkg_train)
    moveOverUnderFlow(sig_test)
    if not bkg_test is None: moveOverUnderFlow(bkg_test)

    # figure out plotrange
    canvas.cd(1)
    yMax = 1e-9
    yMinMax = 1000.
    for h in [sig_train, bkg_train, sig_test, bkg_test]:
        if h is None: continue
        yMax = max(h.GetBinContent(h.GetMaximumBin()), yMax)
        if h.GetBinContent(h.GetMaximumBin()) > 0:
            yMinMax = min(h.GetBinContent(h.GetMaximumBin()), yMinMax)

    # draw first hist
    if plotOptions["logscale"]:
        sig_train.GetYaxis().SetRangeUser(yMinMax/10000, yMax*10)
        canvas.SetLogy()
    else:
        sig_train.GetYaxis().SetRangeUser(0, yMax*1.5)
    sig_train.GetXaxis().SetTitle(canvasName)

    option = "histo"
    sig_train.DrawCopy(option+"E0")

    # draw the other histograms
    if not bkg_train is None: bkg_train.DrawCopy(option+"E0 same")
    if not bkg_test is None: bkg_test.DrawCopy("E0 same")
    sig_test.DrawCopy("E0 same")

    # redraw axis
    canvas.cd(1)
    sig_train.DrawCopy("axissame")

    return canvas


def drawHistsOnCanvas(sigHists, bkgHists, plotOptions, canvasName, realDataHists=None, dataMcRatio=True, displayname=None,logoption=False, workdir=""):
    if not displayname:
        displayname=canvasName
    if sigHists:
        if not isinstance(sigHists, list):
            sigHists = [sigHists]
    if not isinstance(bkgHists, list):
        bkgHists = [bkgHists]
    if realDataHists:
        if not isinstance(realDataHists, list):
            realDataHists = [realDataHists]

    canvas = getCanvas(canvasName, ratiopad=plotOptions["ratio"], dataMcRatio=dataMcRatio)

    # move over/underflow bins into plotrange
    for h in bkgHists:
        moveOverUnderFlow(h)
    if sigHists:
        for h in sigHists:
            moveOverUnderFlow(h)
    if realDataHists:
        for h in realDataHists:
            moveOverUnderFlow(h)

    # stack Histograms
    bkgHists = [bkgHists[len(bkgHists)-1-i] for i in range(len(bkgHists))]
    for i in range(len(bkgHists)-1, 0, -1):
        bkgHists[i-1].Add(bkgHists[i])

    # figure out plotrange
    canvas.cd(1)
    yMax = 1e-9
    yMinMax = 1000.
    for h in bkgHists:
        yMax = max(h.GetBinContent(h.GetMaximumBin()), yMax)
        if h.GetBinContent(h.GetMaximumBin()) > 0:
            yMinMax = min(h.GetBinContent(h.GetMaximumBin()), yMinMax)

    if realDataHists:
        for h in realDataHists:
            yMax = max(h.GetBinContent(h.GetMaximumBin()), yMax)
            if h.GetBinContent(h.GetMaximumBin()) > 0:
                yMinMax = min(h.GetBinContent(h.GetMaximumBin()), yMinMax)
    if sigHists:
        for h in sigHists:
            yMax = max(h.GetBinContent(h.GetMaximumBin()), yMax)
            if h.GetBinContent(h.GetMaximumBin()) > 0:
                yMinMax = min(h.GetBinContent(h.GetMaximumBin()), yMinMax)


    # draw the first histogram
    if sigHists:
        firstHist = sigHists[0]
    else:
        firstHist = bkgHists[0]
    if plotOptions["logscale"] or logoption:
        firstHist.GetYaxis().SetRangeUser(yMinMax/10000, yMax*10)
        canvas.SetLogy()
    else:
        firstHist.GetYaxis().SetRangeUser(0, yMax*1.5)
    firstHist.GetXaxis().SetTitle(displayname)

    option = "histo"
    firstHist.DrawCopy(option+"E0")

    # draw the other histograms
    for h in bkgHists[1:]:
        h.DrawCopy(option+"same")

    canvas.cd(1)
    # redraw axis
    firstHist.DrawCopy("axissame")


    # draw signal histograms
    if sigHists:
        for sH in sigHists:
            # draw signal histogram
            sH.DrawCopy(option+" E0 same")

    #draw real data histograms
    if realDataHists:
        for rD in realDataHists:
            rD.DrawCopy(option+'E0 same')


    ### plot ratio between real Data and simulated data in the analysis AtlasOpenData2012
    if dataMcRatio:

        #hardcoded histogram combination
        canvas.cd(2)
        realDataHist = realDataHists[0]

        MCHist = bkgHists[0]        #the first bkgHist is enoug because they are stacked on each other
        # if len(sigHists) == 1:
        #     MCHist.Add(sigHists[0])
    
       
        realDataHist.Divide(MCHist)
        realDataHist.GetYaxis().SetTitle("Daten/MC")
        realDataHist.SetMaximum(1.4)
        realDataHist.SetMinimum(0.6)
        # for i in range(realDataHist.GetNbinsX()+1):    #make errorbars invisible
        #     # realDataHist.SetBinContent(i, 1)
        #     realDataHist.SetBinError(i, 0.000001)
        realDataHist.Draw("P")

        # save bin information of data/MC histogram to text file
        if workdir:
            # get bin content
            nbins = realDataHist.GetXaxis().GetNbins()
            bin_contents = []
            for i in range(nbins):
                bin_content = realDataHist.GetBinContent(i+1)
                bin_contents.append(bin_content)

            #save to file
            str_to_file = str(workdir)+"/data_mc_ratios.csv"
            if not os.path.isfile(str_to_file):
                df = pd.DataFrame()
                df[canvasName] = bin_contents
                df.to_csv(str_to_file, index=False)
            else:
                df1 = pd.read_csv(str_to_file)
                df2 = pd.DataFrame()
                df2[canvasName] = bin_contents
                df = pd.concat([df1, df2], axis=1)
                df.to_csv(str_to_file, index=False)

            # LEGACY saving
            # file = open(str(workdir)+"/data_mc_ratios.txt","a")
            # file.write(canvasName+"\n")
            # file.write(str(bin_contents)+"\n") 
            # file.close()








    if plotOptions["ratio"]:
        canvas.cd(2)
        line = sigHists[0].Clone()
        line.Divide(sigHists[0])
        line.GetYaxis().SetRangeUser(0.5,1.5)
        line.GetYaxis().SetTitle(plotOptions["ratioTitle"])

        line.GetXaxis().SetLabelSize(line.GetXaxis().GetLabelSize()*2.4)
        line.GetYaxis().SetLabelSize(line.GetYaxis().GetLabelSize()*2.2)
        line.GetXaxis().SetTitle(displayname)

        line.GetXaxis().SetTitleSize(line.GetXaxis().GetTitleSize()*3)
        line.GetYaxis().SetTitleSize(line.GetYaxis().GetTitleSize()*2.5)

        line.GetYaxis().SetTitleOffset(0.5)
        line.GetYaxis().SetNdivisions(505)
        for i in range(line.GetNbinsX()+1):
            line.SetBinContent(i, 1)
            line.SetBinError(i, 1)
        line.SetLineWidth(1)
        line.SetLineColor(ROOT.kBlack)
        line.DrawCopy("histo")
        # ratio plots
        for sigHist in sigHists:
            ratioPlot = sigHist.Clone()
            ratioPlot.Divide(bkgHists[0])
            ratioPlot.SetTitle(displayname)
            ratioPlot.SetLineColor(sigHist.GetLineColor())
            ratioPlot.SetLineWidth(1)
            ratioPlot.SetMarkerStyle(20)
            ratioPlot.SetMarkerColor(sigHist.GetMarkerColor())
            ROOT.gStyle.SetErrorX(0)
            ratioPlot.DrawCopy("sameP")
        canvas.cd(1)

    canvas.cd(1)
    return canvas



# ===============================================
# GENERATE CANVAS AND LEGENDS
# ===============================================
def getCanvas(name, ratiopad = False, dataMcRatio=False):
    #function changed for the atlas OpenData 2012 bachelor thesis
    if ratiopad or dataMcRatio:                         
        canvas = ROOT.TCanvas(name, name, 1024, 1300)
        canvas.Divide(1,2)
        canvas.cd(1).SetPad(0.,0.3,1.0,1.0)
        canvas.cd(1).SetTopMargin(0.2)
        canvas.cd(1).SetBottomMargin(0.05)

        canvas.cd(2).SetPad(0.,0.0,1.0,0.3)
        canvas.cd(2).SetTopMargin(0.1)
        canvas.cd(2).SetBottomMargin(0.4)

        canvas.cd(1).SetRightMargin(0.05)
        canvas.cd(1).SetLeftMargin(0.22)
        canvas.cd(1).SetTicks(1,1)

        canvas.cd(2).SetRightMargin(0.05)
        canvas.cd(2).SetLeftMargin(0.22)
        canvas.cd(2).SetTicks(1,1)

        canvas.cd(2).SetGrid()
    else:
        canvas = ROOT.TCanvas(name, name, 1024, 768)
        canvas.SetTopMargin(0.07)
        canvas.SetBottomMargin(0.15)
        canvas.SetRightMargin(0.05)
        canvas.SetLeftMargin(0.15)
        canvas.SetTicks(1,1)

    return canvas

def getLegend():
    legend=ROOT.TLegend(0.70,0.53,0.95,0.75)
    legend.SetBorderSize(0);
    legend.SetLineStyle(0);
    legend.SetTextFont(42);
    legend.SetTextSize(0.05);
    legend.SetFillStyle(0);
    return legend

def saveCanvas(canvas, path):
    canvas.SaveAs(path)
    canvas.SaveAs(path.replace(".pdf",".png"))
    canvas.Clear()


# ===============================================
# PRINT STUFF ON CANVAS
# ===============================================
def printLumi(pad, lumi = 41.5, ratio = False, twoDim = False):
    pass
    # if lumi == 0.: return

    # lumi_text = str(lumi)+" fb^{-1} (13 TeV)"

    # pad.cd(1)
    # l = pad.GetLeftMargin()
    # t = pad.GetTopMargin()
    # r = pad.GetRightMargin()
    # b = pad.GetBottomMargin()

    # latex = ROOT.TLatex()
    # latex.SetNDC()
    # latex.SetTextColor(ROOT.kBlack)

    # if twoDim:  latex.DrawLatex(l+0.40,1.-t+0.01,lumi_text)
    # elif ratio: latex.DrawLatex(l+0.60,1.-t+0.04,lumi_text)
    # else:       latex.DrawLatex(l+0.53,1.-t+0.02,lumi_text)

def printCategoryLabel(pad, catLabel, ratio = False):
    pass
    # pad.cd(1)
    # l = pad.GetLeftMargin()
    # t = pad.GetTopMargin()
    # r = pad.GetRightMargin()
    # b = pad.GetBottomMargin()

    # latex = ROOT.TLatex()
    # latex.SetNDC()
    # latex.SetTextColor(ROOT.kBlack)

    # if ratio:   latex.DrawLatex(l+0.07,1.-t-0.04, catLabel)
    # else:       latex.DrawLatex(l+0.02,1.-t-0.06, catLabel)

def printROCScore(pad, ROC, ratio = False):
    pad.cd(1)
    l = pad.GetLeftMargin()
    t = pad.GetTopMargin()
    r = pad.GetRightMargin()
    b = pad.GetBottomMargin()

    text = "ROC-AUC = {:.3f}".format(ROC)

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)

    if ratio:   latex.DrawLatex(l+0.05,1.-t+0.04, text)
    else:       latex.DrawLatex(l,1.-t+0.02, text)

def printPrivateWork(pad, ratio = False, twoDim = False, nodePlot = False):
    pad.cd(1)
    l = pad.GetLeftMargin()
    t = pad.GetTopMargin()
    r = pad.GetRightMargin()
    b = pad.GetBottomMargin()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextColor(ROOT.kBlack)
    latex.SetTextSize(0.04)

    text = "CMS private work"

    if nodePlot:    latex.DrawLatex(l+0.57,1.-t+0.01, text)
    elif twoDim:    latex.DrawLatex(l+0.39,1.-t+0.01, text)
    elif ratio:     latex.DrawLatex(l+0.05,1.-t+0.04, text)
    else:           latex.DrawLatex(l,1.-t+0.01, text)


def moveOverUnderFlow(h):
    # move underflow
    h.SetBinContent(1, h.GetBinContent(0)+h.GetBinContent(1))
    # move overflow
    h.SetBinContent(h.GetNbinsX(), h.GetBinContent(h.GetNbinsX()+1)+h.GetBinContent(h.GetNbinsX()))

    # set underflow error
    h.SetBinError(1, ROOT.TMath.Sqrt(
        ROOT.TMath.Power(h.GetBinError(0),2) + ROOT.TMath.Power(h.GetBinError(1),2) ))
    # set overflow error
    h.SetBinError(h.GetNbinsX(), ROOT.TMath.Sqrt(
        ROOT.TMath.Power(h.GetBinError(h.GetNbinsX()),2) + ROOT.TMath.Power(h.GetBinError(h.GetNbinsX()+1),2) ))


def calculateKSscore(stack, sig):
    return stack.KolmogorovTest(sig)
