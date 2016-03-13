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

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Test-class for the own implemented classifiers
 */
public class Test {

    /**
     * Classifiers
     */
    private BinarySplitDT binarySplitDT;
    private MultiSplitDT multiSplitDT;
    private KNN knn;

    /**
     * Data for amble mistake and remaining mistake
     */
    private ReadData readData;
    private Data dataAll;
    private Data dataPass;

    /**
     * Show results
     */
    private Crossvalidation crossvalidation;
    private ConfusionMatrix confusionMatrix;


    /**
     * Method calculates the label-distribution and tests the own implemented classifier with different classifier values
     * @param patternPathAll Storage path pattern-set
     * @param labelPathAll Storage path label-set
     * @param patternPathPass Storage path pattern-set (amble)
     * @param labelPathPass Storage path label-set (amble)
     */
    public void test(String patternPathAll, String labelPathAll, String patternPathPass, String labelPathPass) {
        readData = new ReadData();

        dataPass = readData.readCSVs(patternPathPass, labelPathPass);
        dataAll = readData.readCSVs(patternPathAll, labelPathAll);

        //Calculates the distribution of the individual label sets
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

        //percent/size of train-data
        double split = 0.7;

        //Select mistake for classifier test
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

            //Calculates the label distribution of the train-set
            System.out.println("Train distribution: ");
            distribution = computeDistribution(trainLabel);
            for (int j = 0; j < distribution.length; j++) {
                System.out.println("Label " + (j + 1) + ": " + distribution[j]);
            }

            //Calculates the label distribution of the test-set
            System.out.println("Test distribution: ");
            distribution = computeDistribution(testLabel);
            for (int j = 0; j < distribution.length; j++) {
                System.out.println("Label " + (j + 1) + ": " + distribution[j]);
            }
            System.out.println();

            //K-Nearest-Neighbor with different classifier values
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

            //Binary decision tree with different classifier values
            for(int b = 0; b < 6; b++) {
                System.out.println("Binary: ");
                //split size of the train-data for the decision rules
                System.out.println("Splitsize: " + (5 + 2*b));
                binarySplitDT = new BinarySplitDT();
                binarySplitDT.train(trainPattern, trainLabel, 50, 20, (5 + 2*b));
                classify = binarySplitDT.classify(testPattern);
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                confusionMatrix.computeTrueFalse(classify, testLabel);
                System.out.println();
            }

            //Multi split decision tree with different classifier values
            for(int b = 0; b < 4; b++) {
                System.out.println("Multi: ");
                //split size of the train-data for the decision rules
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

    /**
     * Method calculates the label distribution
     * @param labels label-set
     * @return label distribution
     */
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
