package Classify;

import DT.MultiSplitDT;

/**
 * Created by MatthiasFuchs on 05.01.16.
 */
public class ThreadMDT extends Thread{
    private double[][] trainPatterns;
    private double[] trainLabels;
    private int mode = 0;
    private int labelIndex = -1;
    private MultiSplitDT multiSplitDT;
    private double[][] classifyPatterns;
    private double[] classifyLabels;

    public ThreadMDT(){
        multiSplitDT = new MultiSplitDT();
    }

    public void setTrainData(double[][] patterns, double[] labels) {
        this.trainPatterns = patterns;
        this.trainLabels = labels;
    }

    public void setClassifyData(double[][] patterns) {
        this.classifyPatterns = patterns;
    }

    public void setMode(int mode, int labelIndex) {
        this.mode = mode; // 0 = classify, 1 = train, 2 = load
        this.labelIndex = labelIndex;
    }

    @Override
    public void run() {
        if(mode == 0) {
            classifyLabels = multiSplitDT.classify(classifyPatterns);
        }
        else if(mode == 1) {
            multiSplitDT.train(trainPatterns, trainLabels, 50, 5, 10);
        }
        else if(mode == 2) {
            multiSplitDT.loadData("multiSplitDT" + labelIndex + ".csv");
        }
    }

    public double[] getClassifyLabels() {
        return classifyLabels;
    }
}
