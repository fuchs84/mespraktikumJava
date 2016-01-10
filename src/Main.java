import Classify.Test;
import WekaAlgorithm.Controller;


/**
 * Created by MatthiasFuchs on 06.11.15.
 */
public class Main {
    private static Test test;
    private static Controller controller;
    public static void main(String[] args)  {
        controller = new Controller();
        String patternPathAll = "/Users/MatthiasFuchs/Desktop/Testdaten/Comma-Sep/konstriertdataweka.csv";
        String labelPathAll = "/Users/MatthiasFuchs/Desktop/Testdaten/Comma-Sep/konstriertdatawekalabel.csv";
        String patternPathPass = "/Users/MatthiasFuchs/Desktop/Testdaten/Comma-Sep/konstriertdatawekapass.csv";
        String labelPathPass = "/Users/MatthiasFuchs/Desktop/Testdaten/Comma-Sep/konstriertdatawekalabelpass.csv";
        controller.init(patternPathAll, labelPathAll, patternPathPass, labelPathPass);
        controller.evaluation("crossValidation");
    }
}
