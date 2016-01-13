import Classify.Test;
import WekaAlgorithm.Controller;

import java.util.ArrayList;


/**
 * Created by MatthiasFuchs on 06.11.15.
 */
public class Main {
    private static Test test;
    private static Controller controller;
    public static void main(String[] args)  {
        long start = System.currentTimeMillis();

        String patternPathAll = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/dataMix.csv";
        String labelPathAll = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/labelMix.csv";
        String patternPathPass = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/dataMixpass.csv";
        String labelPathPass = "/Users/MatthiasFuchs/Desktop/Testdaten/Trainset/labelMixpass.csv";
        controller = new Controller();
        controller.init(patternPathAll, labelPathAll, patternPathPass, labelPathPass);
        controller.evaluation("crossValidation");
        System.out.println("Time: " + (System.currentTimeMillis() - start));
        test = new Test();
        test.testTheBests(patternPathAll, labelPathAll, patternPathPass, labelPathPass);
    }
}
