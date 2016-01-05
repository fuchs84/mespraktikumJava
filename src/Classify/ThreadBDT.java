package Classify;

import DT.BinarySplitDT;

/**
 * Created by MatthiasFuchs on 05.01.16.
 */
public class ThreadBDT extends Thread{
    private double[][] trainPatterns;
    private double[] trainLabels;
    private int mode = 0;
    private int labelIndex = -1;
    private BinarySplitDT binarySplitDT;
    private double[][] classifyPatterns;
    private double[] classifyLabels;

    public ThreadBDT(){
        binarySplitDT = new BinarySplitDT();
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
            classifyLabels = binarySplitDT.classify(classifyPatterns);
        }
        else if(mode == 1) {
            binarySplitDT.train(trainPatterns, trainLabels, 50, 5, 10);
        }
        else if(mode == 2) {
            binarySplitDT.loadData("binarySplitDT" + labelIndex + ".csv");
        }
    }

    public double[] getClassifyLabels() {
        return classifyLabels;
    }
}
