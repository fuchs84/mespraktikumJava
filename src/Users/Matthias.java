package Users;

import DT.BSDT.BinarySplitDT;
import DT.MSDT.MultiSplitDT;
import DT.MSDT.MultiSplitNode;
import KNN.KNN;
import MLP.MLP;
import NaiveBayes.NaiveBayes;
import SelectData.Data;
import SelectData.NWData;
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

        mlp = new MLP();
        int[] hidden = {40};
        //mlp.train(trainPattern, trainLabel, hidden, 0.005, 1000);

        binarySplitDT = new BinarySplitDT();
        //binarySplitDT.train(trainPattern, trainLabel, 20, 3);

        multiSplitDT = new MultiSplitDT();
        //multiSplitDT.train(trainPattern, trainLabel, 3);

        //mlp.saveData();
        mlp.loadData();
        mlp.printWeights();

        //multiSplitDT.saveData();
        multiSplitDT.loadData();

        //binarySplitDT.saveData();
        binarySplitDT.loadData();

        confusionMatrix = new ConfusionMatrix();

        double[] classify = mlp.classify(testPattern);
        confusionMatrix.computeConfusionMatrix(classify, testLabel);
        System.out.println();


        classify = binarySplitDT.classify(testPattern);
        confusionMatrix.computeConfusionMatrix(classify, testLabel);
        System.out.println();

        classify = multiSplitDT.classify(testPattern);
        confusionMatrix.computeConfusionMatrix(classify, testLabel);

    }
}
