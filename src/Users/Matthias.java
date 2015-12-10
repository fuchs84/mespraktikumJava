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
    private BinarySplitDT decisionTree;

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


        decisionTree = new BinarySplitDT();
        //decisionTree.train(trainPattern, trainLabel, 20, 3);

        //decisionTree.saveData();
        decisionTree.loadData();

        System.out.println("classify");
        double[] classify = decisionTree.classify(testPattern);

        confusionMatrix = new ConfusionMatrix();
        confusionMatrix.computeConfusionMatrix(classify, testLabel);


    }
}
