package Users;

import DT.BSDT.BinarySplitDT;
import DT.DecisionTree;
import DT.LMDT.LinearMachineDT;
import DT.LMDT.LinearMachineNode;
import DT.MSDT.MultiSplitDT;
import KNN.KNN;
import MLP.MLP;
import NaiveBayes.NaiveBayes;
import SelectData.Data;
import SelectData.NWData;
import ShowData.ConfusionMatrix;
import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;

/**
 * Created by MatthiasFuchs on 12.11.15.
 */
public class Matthias {
    private MLP mlp;
    private MultiSplitDT decisionTree;
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

        //decisionTree = new MultiSplitDT();

        long startTime = System.currentTimeMillis();
        nb = new NaiveBayes();
        nb.train(trainPattern, trainLabel);

        long stopTime = System.currentTimeMillis();
        System.out.println("Train-Time: " + (stopTime-startTime));

        startTime = System.currentTimeMillis();

        double[] classify = nb.classify(testPattern);

        stopTime = System.currentTimeMillis();
        System.out.println("Test-Time: " + (stopTime-startTime));

        confusionMatrix = new ConfusionMatrix();
        confusionMatrix.computeConfusionMatrix(classify, testLabel);





        //confusionMatrix.computeConfusionMatrix(classify, testLabel);
    }
}
