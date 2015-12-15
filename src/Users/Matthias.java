package Users;

import DT.BinarySplitDT;
import DT.MultiSplitDT;
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


    private NWData nwData;
    private DataOld dataOld;

    private ReadData readData;
    private Data data;
    private Crossvalidation crossvalidation;
    private ConfusionMatrix confusionMatrix;

    public void run() {
        automaticTest();
//        String path = "/Users/MatthiasFuchs/Desktop/selectedData.csv";
//
//        nwData = new NWData();
//        dataOld = nwData.readCSV(path);
//
//        double[][] trainPattern = dataOld.trainPattern;
//        double[] trainLabel = dataOld.trainLabel;
//        double[][] testPattern = dataOld.testPattern;
//        double[] testLabel = dataOld.testLabel;
//
//        knn = new KNN();
//        knn.train(trainPattern, trainLabel);
//
//        nb = new NaiveBayes();
//        nb.train(trainPattern, trainLabel);
//
//        mlp = new MLP();
//        int[] hidden = {40};
//        mlp.train(trainPattern, trainLabel, hidden, 0.005, 1000);
//
//        binarySplitDT = new BinarySplitDT();
//        binarySplitDT.train(trainPattern, trainLabel, 20, 5);
//
//        multiSplitDT = new MultiSplitDT();
//        multiSplitDT.train(trainPattern, trainLabel, 3);
//
//
//        //mlp.saveData();
//        //mlp.loadData();
//        //mlp.printWeights();
//
//        multiSplitDT.saveData();
//        //multiSplitDT.loadData();
//
//        binarySplitDT.saveData();
//        //binarySplitDT.loadData();
//
//        confusionMatrix = new ConfusionMatrix();
//
//
//        System.out.println();
//        System.out.println("MLP: ");
//        double[] classify = mlp.classify(testPattern);
//        confusionMatrix.computeTrueFalse(classify, testLabel);
//        System.out.println();
//
//        System.out.println("Naive Bayes: ");
//        classify = nb.classify(testPattern);
//        confusionMatrix.computeTrueFalse(classify, testLabel);
//        System.out.println();
//
//        System.out.println("KNN: ");
//        classify = knn.classify(10, testPattern, "Manhattan");
//        confusionMatrix.computeTrueFalse(classify, testLabel);
//        System.out.println();
//
//
//        System.out.println("Binary DT: ");
//        classify = binarySplitDT.classify(testPattern);
//        confusionMatrix.computeTrueFalse(classify, testLabel);
//        System.out.println();
//
//        System.out.println("Multi DT: ");
//        classify = multiSplitDT.classify(testPattern);
//        confusionMatrix.computeTrueFalse(classify, testLabel);
    }


    private void automaticTest() {
        String labelPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Schritt/20151127ID005labelspass.csv";
        String patternPath = "/Users/MatthiasFuchs/Desktop/Testdaten/Testdaten_Schritt/20151127ID005featurespass.csv";
        readData = new ReadData();
        data = readData.readCSVs(patternPath, labelPath);

        crossvalidation = new Crossvalidation();
        int split = 4;

        for(int a = 0; a < data.getLabel().length; a++) {
            ArrayList<double[][]> crossPattern = crossvalidation.crossvalidate(data.getPattern(), data.getLabel()[a], split);
            ArrayList<double[]> crossLabel = crossvalidation.crossvalidatelabel(data.getPattern(), data.getLabel()[a], split);

            int splitLength = crossLabel.get(0).length;
            double[][] trainPattern = new double[splitLength * (split - 1)][];
            double[] trainLabel = new double[splitLength * (split - 1)];
            double[][] testPattern = new double[splitLength][];
            double[] testLabel = new double[splitLength];

            for (int b = 0; b < split; b++) {
                int trainSelect = b;
                for (int j = 0; j < splitLength; j++) {
                    testLabel[j] = crossLabel.get(trainSelect)[j];
                    testPattern[j] = crossPattern.get(trainSelect)[j];
                }

                int offset = 0;
                for (int i = 0; i < split; i++) {
                    if (i == trainSelect) {
                        offset = splitLength;
                    } else {
                        for (int j = 0; j < splitLength; j++) {
                            trainLabel[j + i * splitLength - offset] = crossLabel.get(i)[j];
                            trainPattern[j + i * splitLength - offset] = crossPattern.get(i)[j];
                        }
                    }
                }

                System.out.println("Selected Label: " + a);
                System.out.println("Selected Train: " + b);


                nb = new NaiveBayes();
                nb.train(trainPattern, trainLabel);

                knn = new KNN();
                knn.train(trainPattern, trainLabel);


                //mlp = new MLP();
                //int[] hidden = {40};
                //mlp.train(trainPattern, trainLabel, hidden, 0.005, 1000);

                binarySplitDT = new BinarySplitDT();
                binarySplitDT.train(trainPattern, trainLabel, 5, 7);

                multiSplitDT = new MultiSplitDT();
                multiSplitDT.train(trainPattern, trainLabel, 2);


                //mlp.saveData();
                //mlp.loadData();
                //mlp.printWeights();

                multiSplitDT.saveData();
                //multiSplitDT.loadData();

                binarySplitDT.saveData();
                //binarySplitDT.loadData();

                confusionMatrix = new ConfusionMatrix();

                double[] classify; //= mlp.classify(testPattern);
                //confusionMatrix.computeConfusionMatrix(classify, testLabel);
                //System.out.println();

                System.out.println("Naive Bayes: ");
                classify = nb.classify(testPattern);
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                System.out.println();


                System.out.println("KNN: ");
                classify = knn.classify(10, testPattern, "Manhattan");
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                System.out.println();


                System.out.println("Binary: ");
                classify = binarySplitDT.classify(testPattern);
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                System.out.println();

                System.out.println("Multi: ");
                classify = multiSplitDT.classify(testPattern);
                confusionMatrix.computeConfusionMatrix(classify, testLabel);
                System.out.println();
            }
        }

    }
}
