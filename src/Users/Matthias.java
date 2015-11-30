package Users;

import DT.BSDT.BinarySplitDT;
import DT.DecisionTree;
import DT.LMDT.LinearMachineDT;
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
    private LinearMachineDT decisionTree;

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

        decisionTree = new LinearMachineDT();

        //Best Result: 2 (10)
        //decisionTree.train(trainPattern, trainLabel);

        confusionMatrix = new ConfusionMatrix();

        double[] classify; //= decisionTree.classify(testPattern);

        //confusionMatrix.computeConfusionMatrix(classify, testLabel);
        //confusionMatrix.computeTrueFalse(classify, testLabel);

        int[] hiddenLayer = {50, 20};
        mlp = new MLP();
        mlp.train(trainPattern, trainLabel,hiddenLayer, 0.005, 1000);
        classify = mlp.classify(testPattern);

        confusionMatrix.computeConfusionMatrix(classify, testLabel);
    }
}
