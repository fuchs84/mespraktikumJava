package Users;

import DT.BSDT.BinarySplitDT;
import DT.DecisionTree;
import DT.LMDT.LinearMachineDT;
import DT.LMDT.LinearMachineNode;
import DT.MSDT.MultiSplitDT;
import MLP.MLP;
import SelectData.Data;
import SelectData.NWData;
import ShowData.ConfusionMatrix;

/**
 * Created by MatthiasFuchs on 12.11.15.
 */
public class Matthias {
    private MLP mlp;
    private MultiSplitDT decisionTree;


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

        //decisionTree = new MultiSplitDT();

        mlp = new MLP();
        mlp.splitData(trainPattern, 10);

        //Best Result: 2 (10)
        //decisionTree.train(trainPattern, trainLabel, 2);

        //confusionMatrix = new ConfusionMatrix();

        //double[] classify = decisionTree.classify(testPattern);

        //confusionMatrix.computeConfusionMatrix(classify, testLabel);
        //confusionMatrix.computeTrueFalse(classify, testLabel);


        //confusionMatrix.computeConfusionMatrix(classify, testLabel);
    }
}
