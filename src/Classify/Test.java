package Classify;

import DT.BinarySplitDT;
import DT.MultiSplitDT;
import KNN.KNN;
import MLP.MLP;
import NaiveBayes.NaiveBayes;
import SelectData.Crossvalidation;
import SelectData.Data;
import SelectData.FeatureSelection;
import SelectData.ReadData;

import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 19.12.15.
 */
public class Test {
    private MLP mlp;
    private BinarySplitDT binarySplitDT;
    private MultiSplitDT multiSplitDT;

    private KNN knn;
    private NaiveBayes nb;

    private ReadData readData;
    private Data data;
    private Data dataAll;
    private Data dataPass;
    private Crossvalidation crossvalidation;
    private ConfusionMatrix confusionMatrix;
    private FeatureSelection featureSelection;


    public void classifierTest() {
        featureSelection = new FeatureSelection();

        String labelPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Samples/LabelsAllpass.csv";
        String patternPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Samples/dataAllpass.csv";
        readData = new ReadData();
    }

    private boolean isNodePure(double[] labels) {
        for (int i = 0; i < labels.length; i++) {
            if (labels[0] !=labels[i]) {
                return false;
            }
        }
        return true;
    }

    public void preselectionTest() {
        readData = new ReadData();

        String labelPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Samples/LabelsAllpass.csv";
        String patternPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Samples/dataAllpass.csv";
        dataPass = readData.readCSVs(patternPath, labelPath);

        labelPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Samples/LabelsAll.csv";
        patternPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Samples/dataAll.csv";

        dataAll = readData.readCSVs(patternPath, labelPath);

        double[] distribution = computeDistribution(dataPass.getLabel()[0]);
        System.out.println("Selected Label-Set: " + 1);
        for(int j = 0; j < distribution.length; j++) {
            System.out.println("Label " + (j+1) + ": " + distribution[j]);
        }
        System.out.println();

        for(int i = 0; i < dataAll.getLabel().length; i++) {
            distribution = computeDistribution(dataAll.getLabel()[i]);
            System.out.println("Selected Label-Set: " + (i+2));
            for(int j = 0; j < distribution.length; j++) {
                System.out.println("Label " + (j+1) + ": " + distribution[j]);
            }
            System.out.println();
        }

        crossvalidation = new Crossvalidation();
        confusionMatrix = new ConfusionMatrix();
        double[] classify;

        double split = 0.7;
        int[] selectedLabelSets = {9, 10};
        for(int i = 0; i < selectedLabelSets.length; i++) {
            ArrayList<ArrayList> randomData = null;
            ArrayList<double[][]> randomPattern;
            ArrayList<double[]> randomLabel;
            switch (selectedLabelSets[i]) {
                case 1:
                    randomData = crossvalidation.randomDataSplit(dataPass.getPattern(), dataPass.getLabel()[0], split);
                    break;
                case 2:
                    randomData = crossvalidation.randomDataSplit(dataAll.getPattern(), dataAll.getLabel()[0], split);
                    break;
                case 3:
                    randomData = crossvalidation.randomDataSplit(dataAll.getPattern(), dataAll.getLabel()[1], split);
                    break;
                case 4:
                    randomData = crossvalidation.randomDataSplit(dataAll.getPattern(), dataAll.getLabel()[2], split);
                    break;
                case 5:
                    randomData = crossvalidation.randomDataSplit(dataAll.getPattern(), dataAll.getLabel()[3], split);
                    break;
                case 6:
                    randomData = crossvalidation.randomDataSplit(dataAll.getPattern(), dataAll.getLabel()[4], split);
                    break;
                case 7:
                    randomData = crossvalidation.randomDataSplit(dataAll.getPattern(), dataAll.getLabel()[5], split);
                    break;
                case 8:
                    randomData = crossvalidation.randomDataSplit(dataAll.getPattern(), dataAll.getLabel()[6], split);
                    break;
                case 9:
                    randomData = crossvalidation.randomDataSplit(dataAll.getPattern(), dataAll.getLabel()[7], split);
                    break;
                case 10:
                    randomData = crossvalidation.randomDataSplit(dataAll.getPattern(), dataAll.getLabel()[8], split);
                    break;
                case 11:
                    randomData = crossvalidation.randomDataSplit(dataAll.getPattern(), dataAll.getLabel()[9], split);
                    break;
                default:
                    System.out.println("Label-Set nicht vorhanden");
                    break;
            }

            System.out.println("Label-Set: " + selectedLabelSets[i]);

            randomPattern = randomData.get(0);
            randomLabel = randomData.get(1);
            double[][] trainPattern = randomPattern.get(0);
            double[] trainLabel = randomLabel.get(0);
            double[][] testPattern = randomPattern.get(1);
            double[] testLabel = randomLabel.get(1);

            System.out.println("Split sizes: ");
            System.out.println("Train: " + trainPattern.length);
            System.out.println("Test: " + testPattern.length);
            System.out.println("Train distribution: ");
            distribution = computeDistribution(trainLabel);
            for (int j = 0; j < distribution.length; j++) {
                System.out.println("Label " + (j + 1) + ": " + distribution[j]);
            }
            System.out.println("Test distribution: ");
            distribution = computeDistribution(testLabel);
            for (int j = 0; j < distribution.length; j++) {
                System.out.println("Label " + (j + 1) + ": " + distribution[j]);
            }
            System.out.println();

            for (int c = 0; c < 6; c++) {
                System.out.println("KNN: ");
                System.out.println("K: " + (5 + 2*c));
                knn = new KNN();
                knn.train(trainPattern, trainLabel);
                classify = knn.classify((5 + 2*c), testPattern, "Manhattan");
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                confusionMatrix.computeTrueFalse(classify, testLabel);
                System.out.println();
            }

            for(int b = 0; b < 6; b++) {
                System.out.println("Binary: ");
                System.out.println("Splitsize: " + (5 + 2*b));
                binarySplitDT = new BinarySplitDT();
                binarySplitDT.train(trainPattern, trainLabel, 50, 20, (5 + 2*b));
                classify = binarySplitDT.classify(testPattern);
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                confusionMatrix.computeTrueFalse(classify, testLabel);
                System.out.println();
            }

            for(int b = 0; b < 4; b++) {
                System.out.println("Multi: ");
                System.out.println("Splitsize: " + (8 + b*2));
                multiSplitDT = new MultiSplitDT();
                multiSplitDT.train(trainPattern, trainLabel, 20, 40, (8 + b*2));
                classify = multiSplitDT.classify(testPattern);
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                confusionMatrix.computeTrueFalse(classify, testLabel);
                System.out.println();
            }
        }
    }

    public void testTheBests() {
        readData = new ReadData();

        String labelPath = "/Users/MatthiasFuchs/Desktop/Testdaten/ID004/All/LabelsAllkonstruiertepass.csv";
        String patternPath = "/Users/MatthiasFuchs/Desktop/Testdaten/ID004/All/dataAllkonstruiertepass.csv";
        dataPass = readData.readCSVs(patternPath, labelPath);

        labelPath = "/Users/MatthiasFuchs/Desktop/Testdaten/ID004/All/LabelsAllkonstruierte.csv";
        patternPath = "/Users/MatthiasFuchs/Desktop/Testdaten/ID004/All/dataAllkonstruierte.csv";

        dataAll = readData.readCSVs(patternPath, labelPath);

        double[] distribution = computeDistribution(dataPass.getLabel()[0]);
        System.out.println("Selected Label-Set: " + 1);
        for(int j = 0; j < distribution.length; j++) {
            System.out.println("Label " + (j+1) + ": " + distribution[j]);
        }
        System.out.println();

        for(int i = 1; i < dataAll.getLabel().length; i++) {
            distribution = computeDistribution(dataAll.getLabel()[i]);
            System.out.println("Selected Label-Set: " + (i+1));
            for(int j = 0; j < distribution.length; j++) {
                System.out.println("Label " + (j+1) + ": " + distribution[j]);
            }
            System.out.println();
        }

        crossvalidation = new Crossvalidation();

        double[] classify;

        int split = 4;

        int[] splitSizesBinary = {};
        int[] splitSizesMulti = {};
        int[] selectedLabelSets = {1, 2, 5, 7, 8, 9, 10, 11};
        for(int i = 0; i < selectedLabelSets.length; i++) {
            confusionMatrix = new ConfusionMatrix();

            ArrayList<ArrayList> crossData = null;
            ArrayList<double[][]> crossPattern;
            ArrayList<double[]> crossLabel;
            switch (selectedLabelSets[i]) {
                case 1:
                    crossData = crossvalidation.crossvalidate(dataPass.getPattern(), dataPass.getLabel()[0], split);
                    break;
                case 2:
                    crossData = crossvalidation.crossvalidate(dataAll.getPattern(), dataAll.getLabel()[1], split);
                    break;
                case 3:
                    crossData = crossvalidation.crossvalidate(dataAll.getPattern(), dataAll.getLabel()[2], split);
                    break;
                case 4:
                    crossData = crossvalidation.crossvalidate(dataAll.getPattern(), dataAll.getLabel()[3], split);
                    break;
                case 5:
                    crossData = crossvalidation.crossvalidate(dataAll.getPattern(), dataAll.getLabel()[4], split);
                    break;
                case 6:
                    crossData = crossvalidation.crossvalidate(dataAll.getPattern(), dataAll.getLabel()[5], split);
                    break;
                case 7:
                    crossData = crossvalidation.crossvalidate(dataAll.getPattern(), dataAll.getLabel()[6], split);
                    break;
                case 8:
                    crossData = crossvalidation.crossvalidate(dataAll.getPattern(), dataAll.getLabel()[7], split);
                    break;
                case 9:
                    crossData = crossvalidation.crossvalidate(dataAll.getPattern(), dataAll.getLabel()[8], split);
                    break;
                case 10:
                    crossData = crossvalidation.crossvalidate(dataAll.getPattern(), dataAll.getLabel()[9], split);
                    break;
                case 11:
                    crossData = crossvalidation.crossvalidate(dataAll.getPattern(), dataAll.getLabel()[10], split);
                    break;
                default:
                    System.out.println("Label-Set nicht vorhanden");
                    break;
            }

            System.out.println("Label-Set: " + selectedLabelSets[i]);
            crossPattern = crossData.get(0);
            crossLabel = crossData.get(1);

            for(int j = 0; j < split; j++) {
                double[][] trainPattern = new double[crossPattern.get(0).length*(split-1)][];
                double[] trainLabel = new double[crossLabel.get(0).length*(split-1)];
                double[][] testPattern = crossPattern.get(j);
                double[] testLabel = crossLabel.get(j);

                int splitLength = crossPattern.get(j).length;
                int offset = 0;
                for(int k = 0; k < split; k++) {
                    if(k != j) {
                        for(int l = 0; l < splitLength; l++) {
                            trainPattern[l + offset] = crossPattern.get(k)[l];
                            trainLabel[l + offset] = crossLabel.get(k)[l];

                        }
                        offset = offset + splitLength;
                    }
                }
                System.out.println("Selected-Train: " + j + " Train/Test-Size: " + trainLabel.length + "/" + testLabel.length);
                System.out.println("Train distribution: ");
                distribution = computeDistribution(trainLabel);
                for (int l = 0; l < distribution.length; l++) {
                    System.out.println("Label " + (l + 1) + ": " + distribution[l]);
                }
                System.out.println("Test distribution: ");
                distribution = computeDistribution(testLabel);
                for (int l = 0; l < distribution.length; l++) {
                    System.out.println("Label " + (l + 1) + ": " + distribution[l]);
                }
                System.out.println();

                System.out.println("KNN: ");
                knn = new KNN();
                knn.train(trainPattern, trainLabel);
                classify = knn.classify(5, testPattern, "Manhattan");
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                confusionMatrix.computeTrueFalse(classify, testLabel);
                confusionMatrix.resultsKNN(classify, testLabel);
                System.out.println();

                System.out.println("Binary: ");
                binarySplitDT = new BinarySplitDT();
                binarySplitDT.train(trainPattern, trainLabel, 200, 5, 10);
                classify = binarySplitDT.classify(testPattern);
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                confusionMatrix.computeTrueFalse(classify, testLabel);
                confusionMatrix.resultsBinary(classify, testLabel);
                System.out.println();

                System.out.println("Multi: ");
                multiSplitDT = new MultiSplitDT();
                multiSplitDT.train(trainPattern, trainLabel, 100, 10, 10);
                classify = multiSplitDT.classify(testPattern);
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                confusionMatrix.computeTrueFalse(classify, testLabel);
                confusionMatrix.resultsMulti(classify, testLabel);
                System.out.println();
            }
            confusionMatrix.printEntireResults();
        }
    }


    public double[] computeDistribution(double[] labels) {
        int numberOfInstances = labels.length;
        double[] distribution = new double[3];
        for(int i = 0; i < labels.length; i++) {
            if((int)labels[i] == 1) {
                distribution[0]++;
            }
            if((int)labels[i] == 2) {
                distribution[1]++;
            }
            if((int)labels[i] == 3) {
                distribution[2]++;
            }
        }
        for(int i = 0; i < distribution.length; i++) {
            distribution[i] = distribution[i] / (double) numberOfInstances;
        }

        return distribution;
    }
}
