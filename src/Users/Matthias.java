package Users;

import DT.BinarySplitDT;
import DT.LMDT.DataOld;
import DT.LMDT.NWData;
import DT.MultiSplitDT;
import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import KNN.KNN;
import MLP.MLP;
import NaiveBayes.NaiveBayes;
import SelectData.*;
import ShowData.ConfusionMatrix;

import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 12.11.15.
 */
public class Matthias {
    private MLP mlp;
    private BinarySplitDT binarySplitDT;
    private MultiSplitDT multiSplitDT;

    private KNN knn;
    private NaiveBayes nb;

    private ReadData readData;
    private Data data;
    private Crossvalidation crossvalidation;
    private ConfusionMatrix confusionMatrix;


    public void run() {
        automaticTest();
    }


    private void automaticTest() {
        String labelPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Samples/LabelsAll.csv";
        String patternPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Samples/dataAll.csv";
        readData = new ReadData();
        data = readData.readCSVs(patternPath, labelPath);

        double[] distribution;
        for(int i = 0; i < data.getLabel().length; i++) {
            distribution = data.computeDistribution(data.getLabel()[i]);
            System.out.println("Selected Feature: " + i);
            for(int j = 0; j < distribution.length; j++) {
                System.out.println("Label " + (j+1) + ": " + distribution[j]);
            }
            System.out.println();
        }

        crossvalidation = new Crossvalidation();
        int split = 5;

        for(int a = 0; a < data.getLabel().length; a++) {
            if (!isNodePure(data.getLabel()[a])) {
                System.out.println("Selected Label: " + a);
                System.out.println();
                ArrayList<ArrayList> randomData = crossvalidation.randomDataSplit(data.getPattern(), data.getLabel()[a], 0.7);
                ArrayList<double[][]> randomPattern = randomData.get(0);
                ArrayList<double[]> randomLabel = randomData.get(1);

                double[][] trainPattern = randomPattern.get(0);
                double[] trainLabel = randomLabel.get(0);
                double[][] testPattern = randomPattern.get(1);
                double[] testLabel = randomLabel.get(1);

                System.out.println("Split sizes: ");
                System.out.println("Train: " + trainPattern.length);
                System.out.println("Test: " + testPattern.length);
                System.out.println("Train distribution: ");
                distribution = data.computeDistribution(trainLabel);
                for (int j = 0; j < distribution.length; j++) {
                    System.out.println("Label " + (j + 1) + ": " + distribution[j]);
                }
                System.out.println("Test distribution: ");
                distribution = data.computeDistribution(testLabel);
                for (int j = 0; j < distribution.length; j++) {
                    System.out.println("Label " + (j + 1) + ": " + distribution[j]);
                }
                System.out.println();


                confusionMatrix = new ConfusionMatrix();
                double[] classify;

                System.out.println("Naive Bayes: ");
                nb = new NaiveBayes();
                nb.train(trainPattern, trainLabel);
                classify = nb.classify(testPattern);
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                confusionMatrix.computeTrueFalse(classify, testLabel);
                System.out.println();

                for (int c = 1; c < 5; c++) {
                    System.out.println("KNN: ");
                    System.out.println("K: " + 5 * c);
                    knn = new KNN();
                    knn.train(trainPattern, trainLabel);
                    classify = knn.classify(5 * c, testPattern, "Manhattan");
                    confusionMatrix.computeConfusionMatrix(classify, testLabel);
                    confusionMatrix.computeTrueFalse(classify, testLabel);
                    System.out.println();
                }


                for (int c = 1; c < 5; c++) {
                    System.out.println("MLP: ");
                    System.out.println("Number of hidden neurons: " + 20 * c);
                    mlp = new MLP();
                    int[] hidden = {20 * c};
                    double[] learningRate = {0.005, 0.01, 0.02, 0.04, 0.08};
                    for (int d = 0; d < learningRate.length; d++) {
                        System.out.println("LearningRate: " + learningRate[d]);
                        mlp.train(trainPattern, trainLabel, hidden, learningRate[d], 100);
                        classify = mlp.classify(testPattern);
                        confusionMatrix.computeConfusionMatrix(classify, testLabel);
                        confusionMatrix.computeTrueFalse(classify, testLabel);
                        System.out.println();
                    }
                }

                for (int d = 1; d < 4; d++) {
                    for (int c = 0; c < 6; c++) {
                        System.out.println("Binary: ");
                        System.out.println("Splitsize: " + 5 * d);
                        if (c == 0) {
                            System.out.println("Without PCA: ");
                            binarySplitDT = new BinarySplitDT();
                            binarySplitDT.train(trainPattern, trainLabel, 20, 20, 5 * d);
                            classify = binarySplitDT.classify(testPattern);
                            confusionMatrix.computeConfusionMatrix(classify, testLabel);
                            confusionMatrix.computeTrueFalse(classify, testLabel);
                            System.out.println();
                        } else {
                            System.out.println("PCA: " + c * 10);
                            binarySplitDT = new BinarySplitDT();
                            binarySplitDT.train(binarySplitDT.computePCA(trainPattern, c * 10), trainLabel, 50, 20, 5 * d);
                            classify = binarySplitDT.classify(binarySplitDT.usePCA(testPattern));
                            confusionMatrix.computeConfusionMatrix(classify, testLabel);
                            confusionMatrix.computeTrueFalse(classify, testLabel);
                            System.out.println();
                        }
                    }
                }

                for (int c = 0; c < 6; c++) {
                    System.out.println("Multi: ");
                    if (c == 0) {
                        System.out.println("Without PCA: ");
                        multiSplitDT = new MultiSplitDT();
                        multiSplitDT.train(trainPattern, trainLabel, 20, 40);
                        classify = multiSplitDT.classify(testPattern);
                        confusionMatrix.computeConfusionMatrix(classify, testLabel);
                        confusionMatrix.computeTrueFalse(classify, testLabel);
                        System.out.println();
                    } else {
                        System.out.println("PCA: " + 10 * c);
                        multiSplitDT = new MultiSplitDT();
                        multiSplitDT.train(multiSplitDT.computePCA(trainPattern, 10 * c), trainLabel, 20, 40);
                        classify = multiSplitDT.classify(multiSplitDT.usePCA(testPattern));
                        confusionMatrix.computeConfusionMatrix(classify, testLabel);
                        confusionMatrix.computeTrueFalse(classify, testLabel);
                        System.out.println();
                    }
                }
//            ArrayList<ArrayList> crossData = crossvalidation.crossvalidate(data.getPattern(), data.getLabel()[a], split);
//            ArrayList<double[][]> crossPattern = crossData.get(0);
//            ArrayList<double[]> crossLabel = crossData.get(1);
//
//            int splitLength = crossLabel.get(0).length;
//            trainPattern = new double[splitLength * (split - 1)][];
//            trainLabel = new double[splitLength * (split - 1)];
//            testPattern = new double[splitLength][];
//            testLabel = new double[splitLength];
//            for (int b = 0; b < split; b++) {
//                int trainSelect = b;
//                for (int j = 0; j < splitLength; j++) {
//                    testLabel[j] = crossLabel.get(trainSelect)[j];
//                    testPattern[j] = crossPattern.get(trainSelect)[j];
//                }
//
//                int offset = 0;
//                for (int i = 0; i < split; i++) {
//                    if (i == trainSelect) {
//                        offset = splitLength;
//                    } else {
//                        for (int j = 0; j < splitLength; j++) {
//                            trainLabel[j + i * splitLength - offset] = crossLabel.get(i)[j];
//                            trainPattern[j + i * splitLength - offset] = crossPattern.get(i)[j];
//                        }
//                    }
//                }
//
//                System.out.println("Crossvalidation Train: " + b);
//
//            }
            }
        }
    }
    protected boolean isNodePure(double[] labels) {
        for (int i = 0; i < labels.length; i++) {
            if (labels[0] !=labels[i]) {
                return false;
            }
        }
        return true;
    }
}
