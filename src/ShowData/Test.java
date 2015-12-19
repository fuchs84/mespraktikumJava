package ShowData;

import DT.BinarySplitDT;
import DT.MultiSplitDT;
import KNN.KNN;
import MLP.MLP;
import NaiveBayes.NaiveBayes;
import SelectData.Crossvalidation;
import SelectData.Data;
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


    public void classifierTest() {
        String labelPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Samples/LabelsAll.csv";
        String patternPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Samples/dataAll.csv";
        readData = new ReadData();
        data = readData.readCSVs(patternPath, labelPath);
        crossvalidation = new Crossvalidation();


        ArrayList<ArrayList> randomData = crossvalidation.randomDataSplit(data.getPattern(), data.getLabel()[3], 0.7);
        ArrayList<double[][]> randomPattern = randomData.get(0);
        ArrayList<double[]> randomLabel = randomData.get(1);
        double[][] trainPattern = randomPattern.get(0);
        double[] trainLabel = randomLabel.get(0);
        double[][] testPattern = randomPattern.get(1);
        double[] testLabel = randomLabel.get(1);


        confusionMatrix = new ConfusionMatrix();
        double[] classify;

        binarySplitDT = new BinarySplitDT();
        //binarySplitDT.train(trainPattern, trainLabel, 50, 20, 10, 20, 1);

        //binarySplitDT.saveData();
        binarySplitDT.loadData();
        classify = binarySplitDT.classify(testPattern);
        confusionMatrix.computeConfusionMatrix(classify, testLabel);

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

        confusionMatrix = new ConfusionMatrix();
        double[] classify;

        double split = 0.7;
        int[] selectedLabelSets = {1, 3, 4, 6, 8, 9, 10};
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

            for (int c = 5; c < 16; c++) {
                System.out.println("KNN: ");
                System.out.println("K: " + c);
                knn = new KNN();
                knn.train(trainPattern, trainLabel);
                classify = knn.classify(c, testPattern, "Manhattan");
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                confusionMatrix.computeTrueFalse(classify, testLabel);
                System.out.println();
            }

            for(int a = 0; a < 3; a++) {
                for(int b = 5; b < 11; b++) {
                    for(int c = 0; c < 6; c++) {
                        System.out.println("Binary: ");
                        System.out.println("Mode: " + a);
                        System.out.println("Splitsize: " + b);
                        System.out.println("PCA: " + c*10);
                        binarySplitDT = new BinarySplitDT();
                        binarySplitDT.train(trainPattern, trainLabel, 50, 20, b, c*10, a);
                        classify = binarySplitDT.classify(testPattern);
                        confusionMatrix.computeConfusionMatrix(classify, testLabel);
                        confusionMatrix.computeTrueFalse(classify, testLabel);
                        System.out.println();
                    }
                }
            }

            for(int b = 8; b < 13; b++) {
                for (int c = 0; c < 6; c++) {
                    System.out.println("Multi: ");
                    System.out.println("Splitsize: " + b);
                    System.out.println("PCA: " + c * 10);
                    multiSplitDT = new MultiSplitDT();
                    multiSplitDT.train(trainPattern, trainLabel, 20, 40, b, c*10);
                    classify = multiSplitDT.classify(testPattern);
                    confusionMatrix.computeConfusionMatrix(classify, testLabel);
                    confusionMatrix.computeTrueFalse(classify, testLabel);
                    System.out.println();
                }
            }
        }
    }

    public void testTheBests() {


        crossvalidation = new Crossvalidation();
        int split = 10;


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
