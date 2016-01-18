import Classify.Test;
import WekaAlgorithm.Controller;
import WekaAlgorithm.VariationConstants;

import java.util.ArrayList;


/**
 * Created by MatthiasFuchs on 06.11.15.
 */
public class Main {
    private static Test test;
    private static Controller controller;
    private static VariationConstants variationConstants;
    public static void main(String[] args) {
        long start = System.currentTimeMillis();

//        String patternPathAll = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/dataMix.csv";
//        String labelPathAll = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/labelMix.csv";
//        String patternPathPass = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/dataMixpass.csv";
//        String labelPathPass = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/labelMixpass.csv";
//        controller = new Controller();
//        controller.init(patternPathAll, labelPathAll, patternPathPass, labelPathPass);
//        controller.evaluation("crossValidation");
        variationConstants = new VariationConstants();
        try {
            variationConstants.automaticVariation();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("Time: " + (System.currentTimeMillis() - start));
    }
}
