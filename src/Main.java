import Classify.Test;
import KNN.KNN;
import SelectData.Data;
import SelectData.ReadData;
import WekaAlgorithm.Controller;
import WekaAlgorithm.VariationConstants;

import java.util.ArrayList;


/**
 * Main starts the program
 */
public class Main {
    private static Controller controller;
    private static VariationConstants variationConstants;

    private static KNN knn;
    private static Data data;
    private static ReadData readData;

    /**
     * Main-method
     * @param args passing arguments
     */
    public static void main(String[] args) {
        long start = System.currentTimeMillis();

        String patternPathAll = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/dataMix3.csv";
        String labelPathAll = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/labelMix.csv";
        String patternPathPass = "";
        String labelPathPass = "";

        //controller = new Controller();
        //controller.init(patternPathAll, labelPathAll, patternPathPass, labelPathPass);
        //controller.evaluation("crossValidation");

        readData = new ReadData();
        data = readData.readCSVs(patternPathAll, labelPathAll);

        knn = new KNN();
        knn.train(data.getPattern(), data.getLabel()[1]);

        System.out.println("Time: " + (System.currentTimeMillis() - start));
    }
}
