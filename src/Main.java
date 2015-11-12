import MLP.MLP;
import SelectData.NWData;
import SelectData.Data;
import ShowData.ConfusionMatrix;

/**
 * Created by MatthiasFuchs on 06.11.15.
 */
public class Main {
    private static MLP mlp;
    private static NWData nwData;
    private static Data data;
    private static ConfusionMatrix confusionMatrix;
    public static void main(String[] args)  {
        String path = "/Users/MatthiasFuchs/Desktop/selectedData.csv";
        nwData = new NWData();
        data = nwData.readCSV(path);

        double[][] label = data.getLabelForMLP(data.getLabel());
        double[][] pattern = data.getScaledPattern(data.getPattern());

        for (int i = 0; i < pattern.length; i++) {
            System.out.println(pattern[i][0]);
        }

        int nInput = pattern[0].length;
        int nOutput = label[0].length;
        int[] nHiddenLayer =  {20};
        mlp = new MLP(nInput, nOutput, nHiddenLayer);
        mlp.printWeights();
        mlp.train(data.getScaledPattern(data.trainPattern), data.getLabelForMLP(data.trainLabel), 0.005, 1000);
        mlp.printWeights();

        double[][] testPattern = data.getScaledPattern(data.testPattern);
        double[] testLabel = data.testLabel;
        int falseClassified = 0;
        int rightCalssified = 0;
        double [] testResult = new double[testPattern.length];
        for (int h = 0; h < testPattern.length; h++) {
            double[] test = mlp.passNetwork(testPattern[h]);
            testResult[h] = mlp.winner(test);
            if((int)testLabel[h] == testResult[h]) {
                rightCalssified++;
            } else {
                falseClassified++;
            }
        }
        System.out.println("Richtig: " + rightCalssified);
        System.out.println("Falsch: " + falseClassified);

        confusionMatrix = new ConfusionMatrix();
        confusionMatrix.computeConfusionMatrix(testLabel, testResult);
    }
}
