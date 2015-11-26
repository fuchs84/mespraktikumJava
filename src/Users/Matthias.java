package Users;

import DT.DecisionTree;
import MLP.MLP;
import SelectData.Data;
import SelectData.NWData;
import ShowData.ConfusionMatrix;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Created by MatthiasFuchs on 12.11.15.
 */
public class Matthias {
    private MLP mlp;
    private DecisionTree decisionTree;

    private NWData nwData;
    private Data data;
    private ConfusionMatrix confusionMatrix;

    public void run() {
        String path = "/Users/MatthiasFuchs/Desktop/selectedData.csv";
        nwData = new NWData();
        data = nwData.readCSV(path);

        double[][] trainPattern = data.trainPattern;
        double[] trainLabel = data.trainLabel;
        double[][] testPattern = data.testPattern;
        double[] testLabel = data.testLabel;

        decisionTree = new DecisionTree();

        //Best Result: 2 (10)
        decisionTree.train(trainPattern, trainLabel, 2, 2);

        confusionMatrix = new ConfusionMatrix();

        double[] classify = decisionTree.classify(testPattern);

        confusionMatrix.computeConfusionMatrix(classify, testLabel);
        confusionMatrix.computeTrueFalse(classify, testLabel);

//        int nInput = pattern[0].length;
//        int nOutput = label[0].length;
//        int[] nHiddenLayer =  {20};
//        mlp = new MLP(nInput, nOutput, nHiddenLayer);
//        mlp.printWeights();
//        mlp.train(data.getScaledPattern(data.trainPattern), data.getLabelForMLP(data.trainLabel), 0.005, 1000);
//        mlp.printWeights();
//
//        int falseClassified = 0;
//        int rightCalssified = 0;
//        double [] testResult = new double[testPattern.length];
//        for (int h = 0; h < testPattern.length; h++) {
//            double[] test = mlp.passNetwork(testPattern[h]);
//            testResult[h] = mlp.winner(test);
//            if((int)testLabel[h] == testResult[h]) {
//                rightCalssified++;
//            } else {
//                falseClassified++;
//            }
//        }
//        System.out.println("Richtig: " + rightCalssified);
//        System.out.println("Falsch: " + falseClassified);
//
//        confusionMatrix = new ConfusionMatrix();
//        confusionMatrix.computeConfusionMatrix(testLabel, testResult);
    }
}
