package Classify;

import KNN.KNN;

/**
 * Created by MatthiasFuchs on 05.01.16.
 */
public class ThreadKNN extends Thread {
    private double[][] trainPatterns;
    private double[] trainLabels;
    private int mode = 0;
    private KNN knn;
    private double[][] classifyPatterns;
    private double[] classifyLabels;

    public ThreadKNN(){
        knn = new KNN();
    }

    public void setTrainData(double[][] patterns, double[] labels) {
        this.trainPatterns = patterns;
        this.trainLabels = labels;
    }

    public void setClassifyData(double[][] patterns) {
        this.classifyPatterns = patterns;
    }

    public void setMode(int mode) {
        this.mode = mode; // 0 = classify, 1 = train
    }

    @Override
    public void run() {
        if(mode == 0) {
            classifyLabels = knn.classify(5, classifyPatterns, "Manhattan");
        }
        else if(mode == 1) {
            knn.train(trainPatterns, trainLabels);
        }
    }

    public double[] getClassifyLabels() {
        return classifyLabels;
    }
}
